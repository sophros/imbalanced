import math
import random
import sys
import time

import numpy as np


# float 0.0 comparison for random
EPS = 0.001  # 1 per mille
DIVERGE_STOP = 13
MAXMIN_RATIO_MIN_DECLINE = 0.01


def tolerance_equal(feature_cnts, tolerance):
    # not equal if any pairwise delta > tolerance
    for i in range(feature_cnts.shape[0]):
        for j in range(i + 1, feature_cnts.shape[0]):
            if abs(feature_cnts[i] - feature_cnts[j]) > tolerance:
                return False

    return True


def average_mututal_delta(feature_cnts):
    total = 0.0

    cnt = 0
    for i in range(feature_cnts.shape[0]):
        for j in range(feature_cnts.shape[0]):
            # for j in range(i + 1, feature_cnts.shape[0]):
            #     total += abs(feature_cnts[i] - feature_cnts[j])
            if 0 != feature_cnts[j]:
                total += feature_cnts[i] / feature_cnts[j]
                cnt += 1

    return total / cnt


def non_zero_min(values):
    assert (0 < len(values))
    # values are all positive
    min_val = values[0] if values[0] != 0 else sys.maxsize

    for i in range(1, len(values)):
        if 0 != values[i] and values[i] < min_val:
            min_val = values[i]

    return min_val


def get_increase_factor(max_cnt, feature_sum_counts, pos, examples_with_feature, counts):
    # finds a factor that is smaller if the row contains > 1 feature and for other features than considered
    # the factor is smaller
    factor_init = factor = 0.0

    # max_cnt = np.max(feature_sum_counts)

    if 0 != examples_with_feature[pos]:
        factor_init = factor = (max_cnt - feature_sum_counts[pos]) / examples_with_feature[pos]

        for i in range(len(counts)):
            if i != pos and 0 != counts[i]:
                if 0 != examples_with_feature[i]:
                    factor_i = (max_cnt - feature_sum_counts[i]) / examples_with_feature[i]
                    if 0.0 <= factor_i < EPS:  # disable / enable minimal change of counts
                        factor = 0.0  # / EPS
                        break

                    if factor_i < factor:
                        factor = factor_i

    # if 10 * factor < factor_init:
    #     factor = -1
    inc_part, inc_int = math.modf(factor)
    return inc_part, int(inc_int)


def balance_corpus_heuristically(binary_features, tolerance=None):
    examples_cnt = binary_features.shape[0]
    print("Number of examples: ", examples_cnt)
    features_cnt = binary_features.shape[1]
    print("Number of features: ", features_cnt)
    counts = np.zeros((examples_cnt, features_cnt + 1,), np.int8)  # counts[0] = index, count[1] = number of examples
    # counts[:, 0] = np.linspace(0, examples_cnt, examples_cnt, dtype=np.int32)  # add index
    counts[:, 0] = 1  # add initial count (uniformly one example)
    counts[:, 1:] = np.copy(binary_features)

    examples_with_feature = binary_features.sum(axis=0)

    feature_sum_counts = examples_with_feature.copy()
    old_min = min_cnt = non_zero_min(feature_sum_counts)
    old_max = max_cnt = np.max(feature_sum_counts)
    if tolerance is None:
        # calculate from delta or 1%
        tolerance = max((np.max(examples_with_feature) - non_zero_min(examples_with_feature)) // (features_cnt / 2),
                        np.average(examples_with_feature) // 100)
    print(f'\nTolerance (1% of the avg difference between pair-wise counts of features OR max difference / no of features): {tolerance}\n')

    # TODO: sampling from the counts of examples
    # TODO: grouping of similar (~ interchangeable) examples and operating on the groups
    #       (eliminating these that cause issues due to cross-bumping counts of other features)

    avg_mut_delta = 0.0
    break_count = loop_cnt = 0
    prev_maxmin_ratio = max_cnt / min_cnt + 2 * MAXMIN_RATIO_MIN_DECLINE  # so that not to stop immediately
    while not tolerance_equal(feature_sum_counts, tolerance):
        prev_avg_mut_delta = avg_mut_delta
        avg_mut_delta = average_mututal_delta(feature_sum_counts)
        maxmin_ratio = max_cnt / min_cnt

        sum_all = np.sum(feature_sum_counts)
        print(f'\n{loop_cnt}:\tdelta: {max_cnt - min_cnt} (avg: {avg_mut_delta}) (delta increase: {break_count} / {DIVERGE_STOP})')
        print(f'\tmin: {min_cnt} ({min_cnt*100 / sum_all} %)\n\tmax: {max_cnt} ({max_cnt*100 / sum_all} %)\nmax/min: {maxmin_ratio}\n')
        print(feature_sum_counts)

        if prev_avg_mut_delta == avg_mut_delta:
            print('Over the last iteration there was no improvement in the average pair-wise differences between feature counts. Stopping.')
            break

        if prev_maxmin_ratio < maxmin_ratio + MAXMIN_RATIO_MIN_DECLINE:
            print(f"Changes of max/min feature counts are too low to attempt to improve further (below {MAXMIN_RATIO_MIN_DECLINE}). Stopping.")
            print(f'previous: {prev_maxmin_ratio}\ncurrent: {maxmin_ratio}')
            break
        else:
            prev_maxmin_ratio = maxmin_ratio

        # START of the calculations
        # indexes sorted in the order of counts
        order_of_adding = [(pos, cnt) for pos, cnt in sorted(enumerate(feature_sum_counts), key=lambda a: a[1])]
        additions_feature_sum_counts = np.zeros(feature_sum_counts.shape, dtype=np.int64)

        for pos, cnt in order_of_adding:
            epos = pos + 1  # it is possible to make all this more optimal by keeping index and counts of examples at the end not at the beginning
            if 0 != examples_with_feature[pos]:
                # what should be the increase factor for the examples with features pos
                increase_per_example = (max_cnt - feature_sum_counts[pos]) / examples_with_feature[pos]
                if increase_per_example > 0.0:
                    inc_part, inc_int = math.modf(increase_per_example)
                    # print(increase_per_example)
                    inc_int = int(inc_int)

                    for i in range(examples_cnt):
                        if counts[i, epos] != 0:  # i-th example exhibits pos-th feature
                            # the approach below does not converge...
                            # if 0 != examples_with_feature[pos]:
                            #     inc_part, inc_int = get_increase_factor(max_cnt, feature_sum_counts, pos, examples_with_feature, counts[i, 1:])

                            delta = int(inc_int) // 2

                            # randomly selecting the given example with probability proportional to the inc_part
                            # inc_part > EPS ensures that we are not calling expensive random() when the chances are v. small.
                            if inc_part >= EPS and random.random() <= inc_part:
                                delta += 1

                            if delta > 0:
                                counts[i, 0] += delta
                                # feature_sum_counts += counts[i, 1:] * delta  # increase counts only for the delta added
                                additions_feature_sum_counts += counts[i, 1:] * delta  # increase counts only for the delta added

        # doing the update of values once per cycle of going through all of the features to eliminate an effect of increase of the max when
            # features are interdependent and bumping counts for one feature bumps for another as well.
        feature_sum_counts += additions_feature_sum_counts
        # = counts[:, 1:].sum(axis=0)

        # if min_cnt > 10 * examples_cnt:  # try approximate factoring of counts
        #     refactor = (max_cnt // min_cnt) * np.min(counts[:, 0])
        #     print(f'\n** Refactoring counts ** {refactor}\n')
        #
        #     counts[:, 0] = np.ceil(counts[:, 0] / refactor)

        # one of the success criteria - is the count gap widening iteration to iteration
        min_cnt = non_zero_min(feature_sum_counts)
        max_cnt = np.max(feature_sum_counts)

        if (old_max - old_min) < (max_cnt - min_cnt):
            break_count += 1

        if break_count > DIVERGE_STOP:
            print('Heuristic started creating a divergent solution. Stopping.')
            break

        old_max, old_min = min_cnt, max_cnt

        loop_cnt += 1

    print("\n\nSOLUTION:")
    print(feature_sum_counts)
    return counts[:, 0], feature_sum_counts


if __name__ == '__main__':
    # balance_corpus_by_simplex()

    test_array = 2500 * [1, ]
    test_array.extend(20000 * [0, ])
    # test_array.extend(250000 * [1, ])
    # simulate imbalance of the dataset:
    test_array.extend(np.random.randint(low=100, high=1000, size=500))
    test_array.extend(np.random.randint(low=2, high=10, size=2000))
    test_array = np.array(test_array)
    np.random.shuffle(test_array)

    T0 = time.time()
    # balance_corpus_heuristically(test_array.reshape(50000, 50))
    balance_corpus_heuristically(test_array.reshape(500, 50))

    print('Total time: ({0:3.2f} s).'.format(time.time() - T0))
