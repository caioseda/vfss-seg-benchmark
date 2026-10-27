'''
O split do protocolo do artigo (`src/data/splits.py`) e o modo `split_mode: protocol` do dataset.

O que precisa ficar preso: contagens EXATAS de pacientes por particao, nenhum paciente em duas
particoes, os folds equilibrados por estrato, e -- o mais importante -- que o teste INCA nao seja
alcancavel por nenhum run de tuning ou CV. Os testes sobre o CSV real do INCA sao pulados quando o
dataset nao esta em disco.

Roda com:  python -m unittest tests.test_protocol_split -v
'''

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.splits import (
    assign_protocol_split,
    load_protocol_split,
    make_protocol_split,
    summarize_protocol_split,
)
from src.data.vfss_frame_dataset import split_by_group

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCA_ROOT = Path("/data_ssd/caioseda/data/dataset_inca_all_frames")
FROZEN_SPLIT = PROJECT_ROOT / "configs" / "splits" / "inca-protocolo-s42.csv"


def make_frames(n_multi: int = 20, n_single: int = 80) -> pd.DataFrame:
    '''`n_multi` pacientes com 3 videos de 4 frames e `n_single` com 1 video de 4 frames.'''
    rows, video = [], 0
    for patient in range(n_multi + n_single):
        for _ in range(3 if patient < n_multi else 1):
            video += 1
            rows += [{"paciente_id": patient, "video_id": video, "frame_id": f} for f in range(4)]
    return pd.DataFrame(rows)


class MakeProtocolSplitTests(unittest.TestCase):

    def setUp(self):
        self.frames = make_frames()
        self.table = make_protocol_split(self.frames, seed=42)

    def test_patient_counts_are_exact(self):
        counts = self.table.partition.value_counts()
        self.assertEqual((counts["dev"], counts["val"], counts["test"]), (70, 10, 20))

    def test_each_stratum_is_split_proportionally(self):
        multi = self.table[self.table.stratum == "multi_video"].partition.value_counts()
        self.assertEqual((multi["test"], multi["val"], multi["dev"]), (4, 2, 14))

    def test_every_patient_appears_once(self):
        self.assertEqual(len(self.table), self.frames.paciente_id.nunique())
        self.assertFalse(self.table.paciente_id.duplicated().any())

    def test_folds_only_for_dev_and_balanced(self):
        dev = self.table[self.table.partition == "dev"]
        self.assertTrue(self.table[self.table.partition != "dev"].fold.isna().all())
        self.assertFalse(dev.fold.isna().any())
        per_fold = dev.fold.value_counts()
        self.assertEqual(sorted(per_fold.index), [0, 1, 2, 3, 4])
        self.assertEqual(per_fold.sum(), 70)
        self.assertLessEqual(per_fold.max() - per_fold.min(), 1)
        multi_per_fold = dev[dev.stratum == "multi_video"].fold.value_counts()
        self.assertLessEqual(multi_per_fold.max() - multi_per_fold.min(), 1)

    def test_deterministic_for_a_seed_and_different_across_seeds(self):
        pd.testing.assert_frame_equal(self.table, make_protocol_split(self.frames, seed=42))
        other = make_protocol_split(self.frames, seed=43)
        self.assertFalse(self.table.partition.equals(other.partition))

    def test_round_half_up_not_bankers_rounding(self):
        # 15 multi-video patients -> 1.5 val patients: must round to 2, as written in the protocol.
        table = make_protocol_split(make_frames(n_multi=15, n_single=99), seed=42)
        multi = table[table.stratum == "multi_video"].partition.value_counts()
        self.assertEqual((multi["test"], multi["val"]), (3, 2))

    def test_summary_counts_add_up(self):
        summary = summarize_protocol_split(self.table)
        partitions = summary[summary.fold.isna()]
        self.assertEqual(partitions.patients.sum(), 100)
        self.assertEqual(partitions.frames.sum(), len(self.frames))


class AssignProtocolSplitTests(unittest.TestCase):

    def setUp(self):
        self.frames = make_frames()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.csv"
            make_protocol_split(self.frames, seed=42).to_csv(path, index=False)
            self.table = load_protocol_split(path)
        self.partition = self.table.set_index("paciente_id").partition
        self.fold = self.table.set_index("paciente_id").fold

    def _patients(self, df, split):
        return set(df[df.split == split].paciente_id.astype(str))

    def test_tuning_trains_on_fold_zero_only(self):
        df = assign_protocol_split(self.frames, self.table, train_folds=[0])
        train = self._patients(df, "train")
        self.assertEqual(train, set(self.fold[self.fold == 0].index))
        self.assertEqual(self._patients(df, "val"), set(self.partition[self.partition == "val"].index))
        self.assertEqual(self._patients(df, "test"), set())

    def test_cv_holds_out_one_fold(self):
        df = assign_protocol_split(self.frames, self.table, cv_fold=2)
        self.assertEqual(self._patients(df, "test"), set(self.fold[self.fold == 2].index))
        dev = set(self.partition[self.partition == "dev"].index)
        self.assertEqual(self._patients(df, "train"), dev - set(self.fold[self.fold == 2].index))

    def test_inca_test_partition_is_never_assigned(self):
        inca_test = set(self.partition[self.partition == "test"].index)
        for kwargs in ({"train_folds": [0]}, {"cv_fold": 0}, {"cv_fold": 4}, {"train_folds": [0, 1, 2, 3, 4]}):
            df = assign_protocol_split(self.frames, self.table, **kwargs)
            rows = df[df.paciente_id.astype(str).isin(inca_test)]
            self.assertTrue(rows.split.isna().all(), kwargs)

    def test_whole_patients_stay_together(self):
        df = assign_protocol_split(self.frames, self.table, cv_fold=1)
        self.assertTrue((df.groupby("paciente_id").split.nunique(dropna=False) == 1).all())

    def test_overlap_between_train_and_cv_fold_is_refused(self):
        with self.assertRaises(ValueError):
            assign_protocol_split(self.frames, self.table, train_folds=[0, 1], cv_fold=1)

    def test_unknown_fold_is_refused(self):
        with self.assertRaises(ValueError):
            assign_protocol_split(self.frames, self.table, cv_fold=7)

    def test_patient_set_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            assign_protocol_split(self.frames[self.frames.paciente_id != 0], self.table, cv_fold=0)
        extra = pd.concat([self.frames, pd.DataFrame([{"paciente_id": 999, "video_id": 999, "frame_id": 0}])])
        with self.assertRaises(ValueError):
            assign_protocol_split(extra, self.table, cv_fold=0)


@unittest.skipUnless((INCA_ROOT / "inca-video-frame-dataset.csv").is_file(), "dataset INCA ausente")
class IncaProtocolSplitTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.data.vfss_frame_dataset import load_video_frame_dataframe
        cls.frames = load_video_frame_dataframe(INCA_ROOT)

    def test_frozen_split_is_what_the_rule_produces(self):
        '''O CSV versionado tem de ser reproduzivel a partir da regra -- senao alguem o editou a mao.'''
        regenerated = make_protocol_split(self.frames, seed=42)
        regenerated["paciente_id"] = regenerated.paciente_id.astype(str)
        frozen = load_protocol_split(FROZEN_SPLIT)
        pd.testing.assert_frame_equal(regenerated.reset_index(drop=True), frozen, check_dtype=False)

    def test_inca_counts(self):
        frozen = load_protocol_split(FROZEN_SPLIT)
        counts = frozen.partition.value_counts()
        self.assertEqual((counts["dev"], counts["val"], counts["test"]), (80, 11, 23))
        dev = frozen[frozen.partition == "dev"]
        self.assertTrue((dev.fold.value_counts() == 16).all())
        self.assertTrue((dev[dev.stratum == "multi_video"].fold.value_counts() == 2).all())

    def test_x1_seed_split_is_unchanged(self):
        '''Regressao: o split do X1 (`split_mode: seed`) continua 690/99/172 frames.'''
        counts = split_by_group(self.frames, "paciente_id", seed=42, ratios=(0.7, 0.1, 0.2)).split.value_counts()
        self.assertEqual((counts["train"], counts["val"], counts["test"]), (690, 99, 172))

    def test_dataset_refuses_test_without_cv_fold(self):
        from src.data.vfss_frame_dataset import VFSSIncaTest
        with self.assertRaisesRegex(ValueError, "cv_fold"):
            VFSSIncaTest(dataset_path=INCA_ROOT, split_mode="protocol", split_file=FROZEN_SPLIT,
                         train_folds=[0], cv_fold=None, target_variant="binary_all")

    def test_dataset_protocol_mode(self):
        from src.data.vfss_frame_dataset import VFSSIncaTest, VFSSIncaTrain
        frozen = load_protocol_split(FROZEN_SPLIT)
        common = dict(dataset_path=INCA_ROOT, split_mode="protocol", split_file=FROZEN_SPLIT,
                      cv_fold=3, target_variant="binary_all")
        train = VFSSIncaTrain(**common).video_frame_df
        test = VFSSIncaTest(**common).video_frame_df
        fold3 = set(frozen[(frozen.partition == "dev") & (frozen.fold == 3)].paciente_id)
        self.assertEqual(set(test.paciente_id.astype(str)), fold3)
        self.assertFalse(set(train.paciente_id.astype(str)) & fold3)
        self.assertFalse(set(train.paciente_id.astype(str)) & set(frozen[frozen.partition != "dev"].paciente_id))


if __name__ == "__main__":
    unittest.main()
