import os
import numpy as np
from torch.utils.data import Dataset
from shapely.geometry import Polygon

class BlockPolygonDataset(Dataset):
    """Dataset loader for urban block annotations stored as a list of dictionaries."""

    def __init__(self, npy_path, split_indices=None, max_polygon=50, max_vert=50, mask_prob=0.0, text_key=None):
        super().__init__()
        self.records = np.load(npy_path, allow_pickle=True).tolist()
        if isinstance(self.records, dict) and 'data' in self.records:
            self.records = self.records['data']
        self.indices = split_indices if split_indices is not None else list(range(len(self.records)))
        self.max_polygon = max_polygon
        self.max_vert = max_vert
        self.mask_prob = mask_prob
        self.text_key = text_key

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        record = self.records[self.indices[idx]]
        bldg_polys = [item[5] for item in record['bldg_poly_list']]
        if len(bldg_polys) > self.max_polygon:
            new_idx = np.random.randint(0, len(self.indices))
            return self.__getitem__(new_idx)

        blk_poly = record.get('blk_recover_poly')
        minx, miny, maxx, maxy = blk_poly.bounds
        w, h = maxx - minx, maxy - miny
        long_side = max(w, h)
        scale = 255.0 / long_side if long_side > 0 else 1.0
        offset_x = (255.0 - w * scale) / 2.0
        offset_y = (255.0 - h * scale) / 2.0

        def normalize(poly):
            coords = np.asarray(poly.exterior.coords[:-1], dtype=np.float32)
            x = (coords[:, 0] - minx) * scale + offset_x
            y = (coords[:, 1] - miny) * scale + offset_y
            return np.stack([x, y], axis=1)

        polygons = [normalize(p) for p in bldg_polys]
        bboxes = [Polygon(p).bounds for p in bldg_polys]

        polygon_verts = np.zeros([self.max_polygon, self.max_vert, 2], dtype=np.float32)
        verts_indicator = np.zeros([self.max_polygon, self.max_vert], dtype=np.float32)
        gt_num_verts = np.zeros([self.max_polygon], dtype=np.int32)
        polygon_mask = self.generate_polygon_mask(polygons)

        for poly_idx, poly in enumerate(polygons):
            processed_polygon = self.preprocess_polygon(poly)
            num_vert = len(processed_polygon)
            gt_num_verts[poly_idx] = num_vert
            polygon_verts[poly_idx, :num_vert, :] = processed_polygon
            verts_indicator[poly_idx, :num_vert] = 1

        polygon_verts = polygon_verts / 127.5 - 1
        verts_indicator_tensor = verts_indicator[..., None] * 2 - 1
        polygon_verts = np.concatenate([polygon_verts, verts_indicator_tensor], axis=-1)
        mimic_proposal_results = self.get_polygons_bbox_init(polygon_verts, bboxes, gt_num_verts)

        data = {
            'image': np.zeros([3, 256, 256], dtype=np.float32),
            'polygon_verts': polygon_verts,
            'verts_indicator': verts_indicator,
            'num_polygon': len(polygons),
            'num_verts': gt_num_verts,
            'polygon_mask': polygon_mask,
            'mimic_proposal_results': mimic_proposal_results,
        }
        if self.text_key is not None and self.text_key in record:
            data['text'] = record[self.text_key]
        return data

    def generate_polygon_mask(self, polygons):
        mask = np.zeros([self.max_polygon], dtype=np.int32)
        if self.mask_prob > 0:
            num_polys = len(polygons)
            rand_mask = (np.random.random([num_polys]) < self.mask_prob)
            mask[:num_polys] = rand_mask.astype(np.int32)
        return mask

    def preprocess_polygon(self, polygon_verts):
        sort_inds = np.lexsort((polygon_verts[:, 1], polygon_verts[:, 0]))
        start_idx = sort_inds[0]
        sorted_verts = np.concatenate([polygon_verts[start_idx:], polygon_verts[:start_idx]], axis=0)
        if self.compute_winding(sorted_verts) < 0:
            sorted_verts = np.concatenate([sorted_verts[0:1], sorted_verts[-1:0:-1]], axis=0)
        return sorted_verts

    @staticmethod
    def compute_winding(polygon_verts):
        winding_sum = 0
        for idx, vert in enumerate(polygon_verts):
            next_idx = idx + 1 if idx < len(polygon_verts) - 1 else 0
            next_vert = polygon_verts[next_idx]
            winding_sum += (next_vert[0] - vert[0]) * (next_vert[1] + vert[1])
        return winding_sum

    def get_polygons_bbox_init(self, polygons, bboxes, gt_num_verts):
        polygons_init = polygons.copy()
        for poly_idx in range(len(bboxes)):
            num_vert = int(gt_num_verts[poly_idx])
            if num_vert == 0:
                continue
            x1, y1, x2, y2 = bboxes[poly_idx]
            rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            if num_vert <= 4:
                coords = rect[:num_vert]
            else:
                t_vals = np.linspace(0, 4, num_vert, endpoint=False)
                coords = []
                for t in t_vals:
                    seg = int(t)
                    frac = t - seg
                    v0 = rect[seg % 4]
                    v1 = rect[(seg + 1) % 4]
                    coords.append(v0 * (1 - frac) + v1 * frac)
                coords = np.stack(coords, axis=0)
            coords = coords / 127.5 - 1
            polygons_init[poly_idx, :num_vert, :2] = coords
        return polygons_init
