from processors.evaluator import Evaluator
import numpy as np

class Predictor(Evaluator):
    def show_1st_hit(self, model, dataset):
        """
        Show the correction prediction of the 1st hit.
        """
        start_id = 0
        raw_ranks_o = []
        raw_ranks_s = []
        flt_ranks_o = []
        flt_ranks_s = []
        for samples in dataset.batch_iter(self.batchsize, rand_flg=False):
            subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
            ids = np.arange(start_id, start_id+len(samples))

            # TODO: partitioned calculation
            # search objects
            raw_scores = model.cal_scores(subs, rels)
            raw_ranks_o.extend(self.cal_rank(raw_scores, objs))
            # filter
            if self.filtered:
                flt_scores = self.cal_filtered_score_fast(subs, rels, objs, ids, raw_scores)
                flt_ranks_o.extend(self.cal_rank(flt_scores, objs))

            # search subjects
            raw_scores_inv = model.cal_scores_inv(rels, objs)
            raw_ranks_s.extend(self.cal_rank(raw_scores_inv, subs))

            # filter
            if self.filtered:
                flt_scores_inv = self.cal_filtered_score_inv_fast(subs, rels, objs, ids, raw_scores_inv)
                flt_ranks_s.extend(self.cal_rank(flt_scores_inv, subs))

            start_id += len(samples)
        
        return (raw_ranks_o, raw_ranks_s, flt_ranks_o, flt_ranks_s)
