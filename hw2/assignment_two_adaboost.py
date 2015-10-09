# Assignment 2, Part 3: Adaboost
#
# Version 0.1
#
# Thanks to the following students for reporting bugs:


import math
import numpy as np
from assignment_two_svm \
    import evaluate_classifier, print_evaluation_summary


# TASK 3.1
# Complete the function definition below
# Remember to return a function, not the
# sign, feature, threshold triple
def weak_learner(instances, labels, dist):

    """ Returns the best 1-d threshold classifier.

    A 1-d threshold classifier is of the form

    lambda x: s*x[j] < threshold

    where s is +1/-1,
          j is a dimension
      and threshold is real number in [-1, 1].

    The best classifier is chosen to minimize the weighted misclassification
    error using weights from the distribution dist.

    """
    m = len(labels)
    n = len(instances[0])
    s = None
    j_star = None
    theta = None
      
    err_pos = np.zeros([m, n])
    err_neg = np.zeros([m, n])
    
    for i in xrange(m):
        for j in xrange(n):
            for k in xrange(m):
                err_pos[k, j] += ((instances[i,j] < instances[k,j]) != \
                                   labels[i]) * dist[i]
                err_neg[k, j] += ((-1 * instances[i,j] < instances[k,j]) != \
                                   labels[i] ) * dist[i]
    
    if np.amin(err_neg) < np.amin(err_pos):
        s = -1
        j_star = np.unravel_index(err_neg.argmin(), err_neg.shape)[1]
        theta = instances[np.unravel_index(err_neg.argmin(), err_neg.shape)]
    else:
        s = 1
        j_star = np.unravel_index(err_pos.argmin(), err_pos.shape)[1]
        theta = instances[np.unravel_index(err_pos.argmin(), err_pos.shape)]
    
    return lambda x: (s * x[j_star]) < theta
    

# TASK 3.2
# Complete the function definition below
def compute_error(h, instances, labels, dist):

    """ Returns the weighted misclassification error of h.

    Compute weights from the supplied distribution dist.
    """
    n = len(instances)    
    error_vec = np.empty(n)
    
    for i in xrange(n):
        error_vec[i] = (h(instances[i]) != labels[i]) * 1
    
    return dist.dot(error_vec)


# TASK 3.3
# Implement the Adaboost distribution update
# Make sure this function returns a probability distribution
def update_dist(h, instances, labels, dist, alpha):

    """ Implements the Adaboost distribution update. """

    n = len(instances)
    new_dist = np.empty(n)
    
    for i in xrange(n):
        if h(instances[i]) == labels[i]:
            new_dist[i] = dist[i] * np.exp(-alpha)
        else:
            new_dist[i] = dist[i] * np.exp(alpha)       
    
    new_dist = new_dist / sum(new_dist)
    
    return new_dist


def run_adaboost(instances, labels, weak_learner, num_iters=20):

    n, d = instances.shape
    n1 = labels.size

    if n1 != n:
        raise Exception('Expected same number of labels as no. of rows in \
                        instances')

    alpha_h = []

    dist = np.ones(n)/n

    for i in range(num_iters):

        print "Iteration: %d" % i
        h = weak_learner(instances, labels, dist)

        error = compute_error(h, instances, labels, dist)

        if error > 0.5:
            break

        alpha = 0.5 * math.log((1-error)/error)

        dist = update_dist(h, instances, labels, dist, alpha)

        alpha_h.append((alpha, h))

    # TASK 3.4
    # return a classifier whose output
    # is an alpha weighted linear combination of the weak
    # classifiers in the list alpha_h
    def classifier(point):
        """ Classifies point according to a classifier combination.

        The combination is stored in alpha_h.
        """
        alphas = np.array(alpha_h)[:, 0]
        hs = np.array(alpha_h)[:, 1]        
        
        n = len(alphas)
        hs_arr = np.empty(n)
        
        for i in xrange(n):
            hs_arr[i] = hs[i](point)
        
        return (alphas.dot(2 * hs_arr - 1) > 0) * 1
        
        
    

    return classifier


def main():
    data_file = 'ionosphere.data'

    data = np.genfromtxt(data_file, delimiter=',', dtype='|S10')
    instances = np.array(data[:, :-1], dtype='float')
    labels = np.array(data[:, -1] == 'g', dtype='int')

    n, d = instances.shape
    nlabels = labels.size

    if n != nlabels:
        raise Exception('Expected same no. of feature vector as no. of labels')

    train_data = instances[:200]  # first 200 examples
    train_labels = labels[:200]  # first 200 labels

    test_data = instances[200:]  # example 201 onwards
    test_labels = labels[200:]  # label 201 onwards

    print 'Running Adaboost...'
    adaboost_classifier = run_adaboost(train_data, train_labels, weak_learner)
    print 'Done with Adaboost!\n'

    confusion_mat = evaluate_classifier(adaboost_classifier, test_data,
                                        test_labels)
    print_evaluation_summary(confusion_mat)

if __name__ == '__main__':
    main()