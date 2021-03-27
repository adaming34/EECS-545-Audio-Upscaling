from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import os

# Only designed for a single voice
class VCTKCroppedDataSet(Dataset):
    """Data from VCTK-Corpus-0.92"""

    def __init__(self, root_dir): # Can possible add transforms later to randomize data better
        """
        Args:
            root_dir (string): Directory of the voice which has the /low_res and /high_res directories
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.low_res_dir = root_dir + '/low_res'
        self.high_res_dir = root_dir + '/high_res'
        self.low_res_files = os.listdir(self.low_res_dir)
        self.high_res_files = os.listdir(self.high_res_dir)

        # ASSERT that the lists idx matches

    def __len__(self):
        return len(self.low_res_files)

    def __getitem__(self, idx):
        low_res = torchaudio.load(self.low_res_dir + '/'+ self.low_res_files[idx])
        high_res = torchaudio.load(self.high_res_dir + '/'+ self.high_res_files[idx])

        sample = {'low_res': low_res[0], 'high_res': high_res[0]}

        # if self.transform:
        #     sample = self.transform(sample)
        return sample

#  JUST TO TEST IT
def main():
    vctk_p255 = VCTKCroppedDataSet(root_dir='data/p225')
    data_loader = DataLoader(vctk_p255, batch_size=1, shuffle=True)

    for data in data_loader:
        input = data["low_res"]
        target = data["high_res"]
        assert input.shape == target.shape

if __name__ == '__main__':
    main()
