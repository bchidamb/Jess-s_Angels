#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <numeric>
#include <random>
#include <algorithm>
#include "csv.h"

using namespace std;


struct Dataset {
    vector<int> row; // user id
    vector<int> col; // movie id
    vector<int> val; // rating
};

// Based on https://stackoverflow.com/questions/29375797
template<typename T>
T** Dynamic2DArray(int w, int h) {
    T** arr = new T*[w];
    arr[0] = new T[w * h];
    for (int i = 1; i < w; i++) {
        arr[i] = arr[i-1] + h;
    }
    return arr;
}

class SVD {
    public:
    double mean;
    double *b_u; // user biases
    double *b_m; // movie biases
    double **p; // user embeddings
    double **q; // movie embeddings
    int lf;

    SVD(int n_users, int n_movies, int latent_factors) {
        lf = latent_factors;
        b_u = new double[n_users];
        b_m = new double[n_movies];
        p = Dynamic2DArray<double>(n_users, lf);
        q = Dynamic2DArray<double>(n_movies, lf);

        default_random_engine generator;
        normal_distribution<double> distribution(0.0, 0.1);

        for (int i = 0; i < n_users; i++) {
            b_u[i] = distribution(generator);
            for (int j = 0; j < lf; j++) {
                p[i][j] = distribution(generator);
            }
        }
        for (int i = 0; i < n_movies; i++) {
            b_m[i] = distribution(generator);
            for (int j = 0; j < lf; j++) {
                q[i][j] = distribution(generator);
            }
        }
    }

    double pred_one(int u, int m) {
        double * p_u = p[u];
        double * q_m = q[m];
        double dot = 0.0;
        for (int i = 0; i < lf; i++) {
            dot += p_u[i] * q_m[i];
        }
        return (dot + b_u[u] + b_m[m] + mean);
    }

    void train(Dataset data, int epochs, double lr, double reg) {
        mean = accumulate(data.val.begin(), data.val.end(), 0.0) / data.row.size();
        vector<int> idx;
        for (int i = 0; i < data.row.size(); i++) {
            idx.push_back(i);
        }

        for (int ep = 0; ep < epochs; ep++) {
            cout << "Epoch " << ep << endl;
            random_shuffle(idx.begin(), idx.end());
            for (int i = 0; i < data.row.size(); i++) {
                int u = data.row[idx[i]];
                int m = data.col[idx[i]];
                double dr = pred_one(u, m) - data.val[idx[i]];
                b_u[u] -= lr * (dr + reg * b_u[u]);
                b_m[m] -= lr * (dr + reg * b_m[m]);
                double * p_u = p[u];
                double * q_m = q[m];
                for (int j = 0; j < lf; j++) {
                    double p_old = p_u[j];
                    double q_old = q_m[j];
                    p_u[j] -= lr * (dr * q_old + reg * p_old);
                    q_m[j] -= lr * (dr * p_old + reg * q_old);
                }
            }
        }
    }

    vector<double> predict(Dataset data) {
        vector<double> pred;
        for(int i = 0; i < data.row.size(); i++) {
            pred.push_back(pred_one(data.row[i], data.col[i]));
        }
        return pred;
    }

    double error(Dataset data) {
        vector<double> pred = this->predict(data);
        double SE = 0.0;
        for(int i = 0; i < data.row.size(); i++) {
            SE += pow(pred[i] - data.val[i], 2);
        }
        return (pow(SE / data.row.size(), 0.5));
    }
};

Dataset load_data(string path) {
    Dataset data;
    io::CSVReader<3> in(path);
    in.read_header(io::ignore_extra_column, "User Number", "Movie Number", "Rating");
    
    int r, c, v;
    while(in.read_row(r, c, v)) {
        data.row.push_back(r - 1);
        data.col.push_back(c - 1);
        data.val.push_back(v);
    }

    return data;
}

void save_submission(string model_name, string ordering, string source, vector<double> predictions) {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer,sizeof(buffer),"%b%d%H%M%S", timeinfo);

    string path = "../submissions/" + ordering + "_" + model_name + "_" + source + "_" + buffer + ".pred";

    FILE * file = fopen(path.c_str(), "w");
    for (double p: predictions) {
        fprintf(file, "%.3f\n", p);
    }
    fclose(file);
}

int main(int argc, char *argv[]) {

    cout << "Loading data..." << endl;

    Dataset train_set = load_data("../data/real_mu_train.csv");
    Dataset test_set1 = load_data("../data/real_mu_probe_sorted.csv");
    Dataset test_set2 = load_data("../data/mu_qual.csv");

    cout << "Training model..." << endl;

    clock_t t = clock();

    SVD model(458293, 17770, 500);
    model.train(train_set, 50, 0.005, 0.02);

    double t_delta = (double) (clock() - t) / CLOCKS_PER_SEC;

    printf("Training time: %.2f s\n", t_delta);

    double rmse = model.error(train_set);
    printf("Train RMSE: %.3f\n", rmse);
    rmse = model.error(test_set1);
    printf("Val RMSE: %.3f\n", rmse);
    
    vector<double> predictions = model.predict(test_set1);
    save_submission("svd", "mu", "real_probe", predictions);
    predictions = model.predict(test_set2);
    save_submission("svd", "mu", "qual", predictions);
    
}
