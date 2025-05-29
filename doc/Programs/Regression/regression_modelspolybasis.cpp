# regression_models_poly_gridsearch.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <numeric>
#include <Eigen/Dense>
#include <algorithm>
#include <functional>

using namespace std;
using namespace Eigen;

// === Utility Functions ===
double mean_squared_error(const VectorXd& y_true, const VectorXd& y_pred) {
    return (y_true - y_pred).squaredNorm() / y_true.size();
}

double r2_score(const VectorXd& y_true, const VectorXd& y_pred) {
    double mean_y = y_true.mean();
    double total = (y_true.array() - mean_y).square().sum();
    double residual = (y_true - y_pred).squaredNorm();
    return 1.0 - residual / total;
}

void save_csv(const string& filename, const MatrixXd& X, const VectorXd& y_true, const VectorXd& y_pred) {
    ofstream file(filename);
    file << "X1,X2,...,True Y,Predicted Y\n";
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j)
            file << X(i, j) << (j == X.cols() - 1 ? "," : ",");
        file << y_true(i) << "," << y_pred(i) << "\n";
    }
    file.close();
}

// === Polynomial Expansion ===
MatrixXd polynomial_expand(const MatrixXd& X, int degree) {
    int n = X.rows();
    int d = X.cols();
    vector<VectorXd> terms;
    terms.push_back(VectorXd::Ones(n));  // bias

    for (int deg = 1; deg <= degree; ++deg) {
        function<void(int, int, VectorXi)> generate;
        generate = [&](int pos, int rem_deg, VectorXi powers) {
            if (pos == d) {
                if (rem_deg == 0) {
                    VectorXd term = VectorXd::Ones(n);
                    for (int i = 0; i < d; ++i)
                        term = term.array() * X.col(i).array().pow(powers(i));
                    terms.push_back(term);
                }
                return;
            }
            for (int i = 0; i <= rem_deg; ++i) {
                powers(pos) = i;
                generate(pos + 1, rem_deg - i, powers);
            }
        };
        generate(0, deg, VectorXi::Zero(d));
    }

    MatrixXd result(n, terms.size());
    for (int i = 0; i < terms.size(); ++i)
        result.col(i) = terms[i];
    return result;
}

// === Cross-validation ===
void cross_validate(const MatrixXd& X, const VectorXd& y, int k,
    function<void(const MatrixXd&, const VectorXd&)> fit_func,
    function<VectorXd(const MatrixXd&)> predict_func,
    double& avg_mse, double& avg_r2) {

    int n = X.rows();
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);
    random_shuffle(indices.begin(), indices.end());

    avg_mse = 0.0;
    avg_r2 = 0.0;

    for (int i = 0; i < k; ++i) {
        int start = i * n / k;
        int end = (i + 1) * n / k;

        vector<int> test_idx(indices.begin() + start, indices.begin() + end);
        vector<int> train_idx;
        for (int j = 0; j < n; ++j)
            if (j < start || j >= end) train_idx.push_back(indices[j]);

        MatrixXd X_train(train_idx.size(), X.cols());
        VectorXd y_train(train_idx.size());
        for (int j = 0; j < train_idx.size(); ++j) {
            X_train.row(j) = X.row(train_idx[j]);
            y_train(j) = y(train_idx[j]);
        }

        MatrixXd X_test(test_idx.size(), X.cols());
        VectorXd y_test(test_idx.size());
        for (int j = 0; j < test_idx.size(); ++j) {
            X_test.row(j) = X.row(test_idx[j]);
            y_test(j) = y(test_idx[j]);
        }

        fit_func(X_train, y_train);
        VectorXd y_pred = predict_func(X_test);
        avg_mse += mean_squared_error(y_test, y_pred);
        avg_r2 += r2_score(y_test, y_pred);
    }

    avg_mse /= k;
    avg_r2 /= k;
}

// === Ridge Regression ===
class RidgeRegression {
public:
    VectorXd weights;
    double alpha;

    RidgeRegression(double alpha = 1.0) : alpha(alpha) {}

    void fit(const MatrixXd& X, const VectorXd& y) {
        MatrixXd X_bias(X.rows(), X.cols() + 1);
        X_bias << MatrixXd::Ones(X.rows(), 1), X;
        MatrixXd I = MatrixXd::Identity(X_bias.cols(), X_bias.cols());
        I(0, 0) = 0;
        weights = (X_bias.transpose() * X_bias + alpha * I).ldlt().solve(X_bias.transpose() * y);
    }

    VectorXd predict(const MatrixXd& X) const {
        MatrixXd X_bias(X.rows(), X.cols() + 1);
        X_bias << MatrixXd::Ones(X.rows(), 1), X;
        return X_bias * weights;
    }
};

// === Main ===
int main() {
    // Synthetic data
    int n_samples = 150, n_features = 2;
    MatrixXd X = MatrixXd::Random(n_samples, n_features);
    VectorXd y = 2 + X * VectorXd::LinSpaced(n_features, 1.0, 2.0) + VectorXd::Random(n_samples) * 0.3;

    int best_degree = 1;
    double best_alpha = 0.0;
    double best_mse = 1e9;

    for (int degree : {1, 2, 3}) {
        MatrixXd X_poly = polynomial_expand(X, degree);
        for (double alpha : {0.01, 0.1, 1.0, 10.0}) {
            RidgeRegression model(alpha);
            double avg_mse, avg_r2;
            cross_validate(X_poly, y, 5,
                [&](const MatrixXd& Xtr, const VectorXd& ytr){ model.fit(Xtr, ytr); },
                [&](const MatrixXd& Xte){ return model.predict(Xte); },
                avg_mse, avg_r2);

            cout << "Degree=" << degree << ", Alpha=" << alpha
                 << " -> MSE=" << avg_mse << ", R2=" << avg_r2 << endl;

            if (avg_mse < best_mse) {
                best_mse = avg_mse;
                best_alpha = alpha;
                best_degree = degree;
            }
        }
    }

    cout << "\nBest model: Degree=" << best_degree
         << ", Alpha=" << best_alpha
         << ", MSE=" << best_mse << endl;

    MatrixXd X_poly_best = polynomial_expand(X, best_degree);
    RidgeRegression best_model(best_alpha);
    best_model.fit(X_poly_best, y);
    VectorXd y_pred = best_model.predict(X_poly_best);

    save_csv("predictions_poly_ridge.csv", X, y, y_pred);

    return 0;
}


// g++ regression_models_poly_gridsearch.cpp -o poly_model -I /path/to/eigen ./poly_model
