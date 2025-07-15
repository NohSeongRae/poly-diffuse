import os
import numpy as np
from torch.utils.data import Dataset
from shapely.geometry import Polygon

class UrbanPolygonDataset(Dataset):
    """Dataset loading building polygons and land-lot bounding boxes."""

    def __init__(self, data_dir, split, init_dir=None, rand_aug=False,
                 max_polygon=50, max_vert=50, mask_prob=0.0):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.init_dir = init_dir
        self.rand_aug = rand_aug
        self.max_polygon = max_polygon
        self.max_vert = max_vert
        self.mask_prob = mask_prob

        split_dir = os.path.join(data_dir, split)
        self.files = sorted([
            os.path.join(split_dir, f) for f in os.listdir(split_dir)
            if f.endswith('.npz') or f.endswith('.npy')
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_path = self.files[idx]
        sample = np.load(sample_path, allow_pickle=True)
        polygons = sample['polygons']
        bboxes = sample['bboxes']
        polygons = [np.array(p) for p in polygons]

        if len(polygons) > self.max_polygon:
            new_idx = np.random.randint(0, len(self.files))
            return self.__getitem__(new_idx)

        polygon_verts = np.zeros([self.max_polygon, self.max_vert, 2], dtype=np.float32)
        verts_indicator = np.zeros([self.max_polygon, self.max_vert], dtype=np.float32)
        gt_num_verts = np.zeros([self.max_polygon], dtype=np.int32)

        polygon_mask = self.generate_polygon_mask(polygons)

        for poly_idx, poly_item in enumerate(polygons):
            processed_polygon = self.preprocess_polygon(poly_item)
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
        return data

    def generate_polygon_mask(self, polygons):
        mask = np.zeros([self.max_polygon], dtype=np.int32)
        if 'train' in self.split and self.mask_prob > 0:
            num_polys = len(polygons)
            rand_mask = (np.random.random([num_polys]) < self.mask_prob)
            mask[:num_polys] = rand_mask.astype(np.int32)
        return mask

    def preprocess_polygon(self, polygon_verts):
        sort_inds = np.lexsort((polygon_verts[:, 1], polygon_verts[:, 0]))
        start_idx = sort_inds[0]
        sorted_verts = np.concatenate([polygon_verts[start_idx:], polygon_verts[:start_idx]], axis=0)
        winding = self.compute_winding(sorted_verts)
        if winding < 0:
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
