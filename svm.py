from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy.linalg 
import pickle
import matplotlib.pyplot as plt
import itertools
import cvxopt
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

cvxopt.solvers.options['show_progress'] = False

def linear_kernel(**kwargs):
    def f(x1, x2):
        return np.inner(x1, x2)
    return f

def polynomial_kernel(power, coef, **kwargs):
    def f(x1, x2):
        return (np.inner(x1, x2) + coef)**power
    return f

def rbf_kernel(gamma, **kwargs):
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f


class SupportVectorMachine(object):
    def __init__(self, C=1, kernel=polynomial_kernel, power=4, gamma=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        if not self.gamma:
            self.gamma = 1 / n_features

        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)

        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        G_max = np.identity(n_samples) * -1
        G_min = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((G_max, G_min)))
        h_max = cvxopt.matrix(np.zeros(n_samples))
        h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
        h = cvxopt.matrix(np.vstack((h_max, h_min)))

        print(P)
        print(q)
        print(A)
        print(b)
        print(G)
        print(h)
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
        

        lagr_mult = np.ravel(minimization['x'])
        print(minimization['x'])

        idx = lagr_mult > 1e-7
        ind = np.arange(len(lagr_mult))[idx]
        self.lagr_multipliers = lagr_mult[idx]
        self.support_vectors = X[idx]
        self.support_vector_labels = y[idx]

        self.b = 0
        for n in range(len(self.lagr_multipliers)):
            self.b += self.support_vector_labels[n]
            self.b -= np.sum(self.lagr_multipliers * self.support_vector_labels * kernel_matrix[ind[n], idx])

        self.b /= len(self.lagr_multipliers)
        print(self.b)
        print(self.support_vector_labels)
        print(self.support_vectors)
        print(self.lagr_multipliers)

    def predict_proba(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.lagr_multipliers, self.support_vector_labels, self.support_vectors):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict + self.b

    def predict(self, X):
        return np.sign(self.predict_proba(X))

def main():
    data = pd.read_csv("dataset/dataset_clean_labelled.csv")

    trainX, testX, trainY, testY = train_test_split(data["TWEET"].values.astype(str), data["LABEL"].values.astype(int), test_size=0.2)

    trainY[trainY == 0] = -1
    testY[testY == 0] = -1
    vectorizer = pickle.load(open("model/vectorizer.mdl", 'rb'))

    vector = np.asarray([[0.778, 0.778, 0.778, 0.778, 0.778, 0.778, 0.778], [0, 0.778, 0.778, 0.778, 0.778, 0.778, 0.778]])
    label = np.asarray([1, -1])

    x_train = vectorizer.transform(trainX)

    clf = SupportVectorMachine(kernel=polynomial_kernel, power=4, coef=1)
    clf.fit(vector, label)

    print(clf.predict([[0.778, 0.778, 0.778, 0.778, 0.778, 0.778, 0.778]]))

    #x_test = vectorizer.transform(testX)

    #y_pred = clf.predict(x_test.toarray())
    #print(clf.predict(x_test.toarray()))
    #y_pred = np.argmax(clf.predict(x_test.toarray()))

    '''
    score = {
        'accuracy': accuracy_score(testY, y_pred),
        'precision': precision_score(testY, y_pred),
        'recall': recall_score(testY, y_pred),
        'f1': f1_score(testY, y_pred),
    }
    
    print(score)
    '''

if __name__ == "__main__":
    main()