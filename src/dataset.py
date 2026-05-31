import torch
from torch.utils.data import Dataset

class MultimodalEmbeddingDataset(Dataset):
    """
    Dataset to load precomputed (cached) visual, audio, and teacher video embeddings.
    """
    def __init__(self, file_path=None, data_dict=None):
        if file_path is not None:
            self.data = torch.load(file_path, map_location="cpu")
        elif data_dict is not None:
            self.data = data_dict
        else:
            raise ValueError("Either file_path or data_dict must be provided.")
            
        # Keys expected: 'z_img', 'z_aud', 'v_teacher'
        self.z_img = self.data['z_img'].to(torch.float32)
        self.z_aud = self.data['z_aud'].to(torch.float32)
        self.v_teacher = self.data['v_teacher'].to(torch.float32)
        
        # Verify sizes
        assert len(self.z_img) == len(self.z_aud) == len(self.v_teacher), \
            f"Dimension mismatch in dataset: img={len(self.z_img)}, aud={len(self.z_aud)}, teacher={len(self.v_teacher)}"

    def __len__(self):
        return len(self.z_img)

    def __getitem__(self, idx):
        return {
            'z_img': self.z_img[idx],
            'z_aud': self.z_aud[idx],
            'v_teacher': self.v_teacher[idx]
        }
