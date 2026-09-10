'''
Where the unlabeled frames in a batch come from -- the variable experiment X1B isolates.

The literature's two-stream samplers draw the unlabeled rows from the whole pool because their
benchmarks are collections of independent slices. On video data that is a choice, not a given, and
these tests pin what each choice actually delivers:

  - `same_video` really does pair with the anchor's video (a policy that quietly ignores the anchor
    would still train, and still look reasonable);
  - `temporal_window` respects W, and larger W strictly contains smaller W;
  - the *labeled* stream is byte-identical across policies at a fixed seed -- without that, X1B
    would be comparing policies plus whatever else moved;
  - when a policy cannot be served, the fallback is recorded rather than silent.

Run with:  python -m unittest tests.test_sampling_policies -v
'''

import unittest

import numpy as np

from src.data.samplers import TwoStreamBatchSampler

N_VIDEOS = 3
FRAMES_PER_VIDEO = 20
LABELED_FRAMES = (5, 15)      # per video


def make_index(n_videos=N_VIDEOS, frames_per_video=FRAMES_PER_VIDEO, labeled_frames=LABELED_FRAMES):
    '''Synthetic dataset layout: `n_videos` videos of `frames_per_video` frames, row-major.'''
    video_ids, frame_ids = [], []
    for video in range(1, n_videos + 1):
        for frame in range(1, frames_per_video + 1):
            video_ids.append(video)
            frame_ids.append(frame)

    metadata = {"video_id": np.array(video_ids), "frame_id": np.array(frame_ids)}
    labeled = [i for i, (v, f) in enumerate(zip(video_ids, frame_ids)) if f in labeled_frames]
    unlabeled = [i for i, (v, f) in enumerate(zip(video_ids, frame_ids)) if f not in labeled_frames]
    return metadata, labeled, unlabeled


def build(policy="global", temporal_window=None, seed=0, batch_size=8, labeled_batch_size=4,
          metadata=None, labeled=None, unlabeled=None):
    if metadata is None:
        metadata, labeled, unlabeled = make_index()
    return TwoStreamBatchSampler(
        labeled, unlabeled, batch_size=batch_size, labeled_batch_size=labeled_batch_size,
        seed=seed, unlabeled_policy=policy, temporal_window=temporal_window,
        frame_metadata=metadata,
    ), metadata


def split_batch(batch, labeled_batch_size):
    return batch[:labeled_batch_size], batch[labeled_batch_size:]


class TestSameVideoPolicy(unittest.TestCase):

    def test_every_unlabeled_row_shares_its_anchor_video(self):
        sampler, metadata = build(policy="same_video")
        videos = metadata["video_id"]

        for batch in sampler:
            labeled, unlabeled = split_batch(batch, sampler.labeled_batch_size)
            for slot, index in enumerate(unlabeled):
                anchor = labeled[slot % len(labeled)]
                self.assertEqual(videos[index], videos[anchor],
                                 "unlabeled row does not come from its anchor's video")

    def test_global_policy_ignores_the_anchor(self):
        '''The contrast that makes the test above meaningful: global must NOT track the anchor.'''
        sampler, metadata = build(policy="global", seed=3)
        videos = metadata["video_id"]

        mismatches = 0
        for batch in sampler:
            labeled, unlabeled = split_batch(batch, sampler.labeled_batch_size)
            for slot, index in enumerate(unlabeled):
                anchor = labeled[slot % len(labeled)]
                mismatches += int(videos[index] != videos[anchor])
        self.assertGreater(mismatches, 0, "global policy behaved like same_video")


class TestTemporalWindowPolicy(unittest.TestCase):

    def test_draws_stay_inside_the_window(self):
        for window in (3, 5, 10):
            with self.subTest(W=window):
                sampler, metadata = build(policy="temporal_window", temporal_window=window)
                videos, frames = metadata["video_id"], metadata["frame_id"]

                for batch in sampler:
                    labeled, unlabeled = split_batch(batch, sampler.labeled_batch_size)
                    for slot, index in enumerate(unlabeled):
                        anchor = labeled[slot % len(labeled)]
                        self.assertEqual(videos[index], videos[anchor])
                        self.assertLessEqual(abs(int(frames[index]) - int(frames[anchor])), window,
                                             f"draw outside the +/-{window} window")

    def test_larger_windows_contain_smaller_ones(self):
        '''
        W is a knob on *proximity*; if the reachable sets did not nest it would not be one.

        Asserted on the candidate sets themselves rather than on what sampling happens to visit:
        drawing is random, so a coverage-based version of this test would be flaky for reasons that
        have nothing to do with the property.
        '''
        # Long videos, so that even W=10 leaves frames out of reach -- with 20-frame videos and
        # anchors at 5 and 15, every window from W=5 up already covers the whole video and the
        # nesting would be vacuous.
        metadata, labeled, unlabeled = make_index(frames_per_video=60, labeled_frames=(20, 40))

        reachable = {}
        for window in (3, 5, 10):
            sampler, _ = build(policy="temporal_window", temporal_window=window,
                               metadata=metadata, labeled=labeled, unlabeled=unlabeled)
            candidates = set()
            for anchor in labeled:
                rows, level = sampler._candidates_for(anchor)
                self.assertEqual(level, "temporal_window", "the fixture forced a fallback")
                candidates.update(rows.tolist())
            reachable[window] = candidates

        self.assertTrue(reachable[3] < reachable[5], "W=5 did not strictly contain W=3")
        self.assertTrue(reachable[5] < reachable[10], "W=10 did not strictly contain W=5")
        self.assertTrue(reachable[10] < set(unlabeled), "W=10 already reaches every frame")

    def test_anchor_is_never_drawn_as_its_own_unlabeled_partner(self):
        # A dataset where the anchor frame itself sits in the unlabeled pool.
        metadata, labeled, unlabeled = make_index()
        anchor = labeled[0]
        unlabeled = sorted(unlabeled + [anchor])
        sampler, _ = build(policy="temporal_window", temporal_window=3,
                           metadata=metadata, labeled=labeled, unlabeled=unlabeled)

        for batch in sampler:
            labeled_batch, unlabeled_batch = split_batch(batch, sampler.labeled_batch_size)
            for slot, index in enumerate(unlabeled_batch):
                self.assertNotEqual(index, labeled_batch[slot % len(labeled_batch)],
                                    "a frame was paired with itself: that is not consistency")

    def test_window_requires_a_width(self):
        metadata, labeled, unlabeled = make_index()
        with self.assertRaises(ValueError):
            TwoStreamBatchSampler(labeled, unlabeled, batch_size=8, labeled_batch_size=4,
                                  unlabeled_policy="temporal_window", frame_metadata=metadata)

    def test_video_conditioned_policies_require_metadata(self):
        _, labeled, unlabeled = make_index()
        for policy in ("same_video", "temporal_window"):
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    TwoStreamBatchSampler(labeled, unlabeled, batch_size=8, labeled_batch_size=4,
                                          unlabeled_policy=policy, temporal_window=5)


class TestControlAcrossPolicies(unittest.TestCase):
    '''What makes X1B a comparison of policies rather than of everything at once.'''

    def test_labeled_stream_is_identical_across_policies(self):
        streams = {}
        for policy, window in [("global", None), ("same_video", None),
                               ("temporal_window", 3), ("temporal_window", 10)]:
            sampler, _ = build(policy=policy, temporal_window=window, seed=42)
            streams[(policy, window)] = [batch[:sampler.labeled_batch_size] for batch in sampler]

        reference = streams[("global", None)]
        for key, stream in streams.items():
            self.assertEqual(stream, reference,
                             f"policy {key} changed the labeled stream; the comparison is confounded")

    def test_batch_composition_is_unchanged(self):
        for policy, window in [("global", None), ("same_video", None), ("temporal_window", 5)]:
            with self.subTest(policy=policy):
                sampler, _ = build(policy=policy, temporal_window=window)
                labeled_set = set(sampler.labeled_indices)
                for batch in sampler:
                    self.assertEqual(len(batch), sampler.batch_size)
                    self.assertEqual(sum(index in labeled_set for index in batch),
                                     sampler.labeled_batch_size)


class TestFallbacksAreVisible(unittest.TestCase):
    '''
    A policy that cannot be served must say so. A `temporal_window` run silently falling back is a
    partly-global run wearing a W label, and would be read off the X1B table as if it were not.
    '''

    def test_video_without_unlabeled_frames_falls_back_to_global(self):
        metadata, labeled, unlabeled = make_index()
        videos = metadata["video_id"]
        # Video 1 contributes no unlabeled frames at all.
        unlabeled = [i for i in unlabeled if videos[i] != 1]

        sampler, _ = build(policy="same_video", metadata=metadata, labeled=labeled, unlabeled=unlabeled)
        for _ in sampler:
            pass

        report = sampler.draw_report()
        self.assertGreater(report["global"], 0.0, "fallback happened but was not recorded")
        self.assertLess(report["served_as_requested"], 1.0)
        self.assertAlmostEqual(report["same_video"] + report["global"], 1.0, places=6)

    def test_window_with_no_neighbours_falls_back_to_same_video(self):
        # Unlabeled frames only far from the labeled ones (5 and 15), so W=1 can never be served.
        metadata, labeled, _ = make_index()
        frames = metadata["frame_id"]
        unlabeled = [i for i in range(len(frames)) if i not in set(labeled) and frames[i] in (1, 20)]

        sampler, _ = build(policy="temporal_window", temporal_window=1,
                           metadata=metadata, labeled=labeled, unlabeled=unlabeled)
        for _ in sampler:
            pass

        report = sampler.draw_report()
        self.assertEqual(report["served_as_requested"], 0.0)
        self.assertAlmostEqual(report["same_video"], 1.0, places=6,
                               msg="fallback should stop at same_video, not go straight to global")

    def test_report_is_all_requested_when_the_policy_is_servable(self):
        sampler, _ = build(policy="temporal_window", temporal_window=5)
        for _ in sampler:
            pass
        self.assertAlmostEqual(sampler.draw_report()["served_as_requested"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
