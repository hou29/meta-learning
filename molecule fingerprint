import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.impute import SimpleImputer
from tqdm import tqdm


class FeaturesGeneration:
    # RDKit descriptors -->
    def __init__(self, nbits: int = 1024, long_bits: int = 16384):
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])

        # dictionary
        self.fp_func_dict = {'ecfp0': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits),
                             'ecfp2': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits),
                             'ecfp4': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits),
                             'ecfp6': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits),
                             'fcfp2': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True,
                                                                                      nBits=nbits),
                             'fcfp4': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True,
                                                                                      nBits=nbits),
                             'fcfp6': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True,
                                                                                      nBits=nbits),
                             'lecfp4': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=long_bits),
                             'lecfp6': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=long_bits),
                             'lfcfp4': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True,
                                                                                       nBits=long_bits),
                             'lfcfp6': lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True,
                                                                                       nBits=long_bits),
                             'maccs': lambda m: MACCSkeys.GenMACCSKeys(m),
                             'hashap': lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits),
                             'hashtt': lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m,
                                                                                                                  nBits=nbits),
                             'avalon': lambda m: fpAvalon.GetAvalonFP(m, nbits),
                             'laval': lambda m: fpAvalon.GetAvalonFP(m, long_bits),
                             'rdk5': lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2),
                             'rdk6': lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2),
                             'rdk7': lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2),
                             'rdkDes': lambda m: calc.CalcDescriptors(m)}

    def get_fingerprints(self, smiles_list: list, fp_name: str) -> np.array:
        """获得分子指纹fingerprint

        Args:
            smiles_list: list(smiles序列)
            fp_name: fingerprint分子指纹名称

        Returns:

        """
        fingerprints = []
        not_found = []

        for smi in tqdm(smiles_list):
            try:
                m = Chem.MolFromSmiles(smi)
                fp = self.fp_func_dict[fp_name](m)
                fingerprints.append(np.array(fp))
            except ValueError:
                not_found.append(smi)
                if fp_name == 'rdkDes':
                    tpatf_arr = np.empty(len(fingerprints[0]), dtype=np.float32)
                else:
                    tpatf_arr = np.empty(len(fingerprints[0]), dtype=np.float32)

                fingerprints.append(tpatf_arr)

        if fp_name == 'rdkDes':
            x = np.array(fingerprints)
            ndf = pd.DataFrame.from_records(x)
            [ndf[col].fillna(ndf[col].mean(), inplace=True) for col in ndf.columns]
            x = ndf.iloc[:, 0:].values
            x = x.astype(np.float32)
            x = np.nan_to_num(x)
        else:
            fp_array = (np.array(fingerprints, dtype=object))
            x = np.vstack(fp_array).astype(np.float32)
            imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
            imp_median.fit(x)
            x = imp_median.transform(x)

        final_array = x

        return final_array


if __name__ == '__main__':
    smiles_list = ['CC#CC', 'O=CCO', 'N=C(N)C=O']
    featuresGeneration = FeaturesGeneration()
    features = featuresGeneration.get_fingerprints(smiles_list, fp_name='maccs')
    print(features)
