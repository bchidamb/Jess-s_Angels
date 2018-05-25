#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_set>
#include "csv.h"

/*
 * DONT FORGET: compile with "-Ofast -march=native"
 */

#define n_users 458293
#define n_movies 17770
#define n_bins 30
#define beta 0.4

using namespace std;

class Dataset {
    public:
    vector<int> row; // user id
    vector<int> col; // movie id
    vector<int> val; // rating
    vector<int> day; // date
    vector<int> bin; // date-bin (0 - 29)
    vector<vector<int>> mpu; // movies per user

    // dates per user
    vector<double> mean_date;
    vector<vector<int>> dpu;
    vector<unordered_set<int>> unique_dpu;

    Dataset() {
        vector<vector<int>> temp(n_users, vector<int>(0,0));
        mpu = temp;

        vector<vector<int>> temp2(n_users, vector<int>(0,0));
        dpu = temp2;

        vector<unordered_set<int>> temp3(n_users, unordered_set<int>({}));
        unique_dpu = temp3;
    }
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

    SVD(int latent_factors) {
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
        double dot = 0.0;
        for (int i = 0; i < lf; i++) {
            dot += p[u][i] * q[m][i];
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
                for (int j = 0; j < lf; j++) {
                    double p_old = p[u][j];
                    double q_old = q[m][j];
                    p[u][j] -= lr * (dr * q_old + reg * p_old);
                    q[m][j] -= lr * (dr * p_old + reg * q_old);
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

class SVDpp {
    public:
    double mean;
    double *b_u; // user biases
    double *b_m; // movie biases
    double **p; // user embeddings
    double **q; // movie embeddings
    double **y; // implicit latent factor
    int lf;

    SVDpp(int latent_factors) {
        lf = latent_factors;
        b_u = new double[n_users];
        b_m = new double[n_movies];
        p = Dynamic2DArray<double>(n_users, lf);
        q = Dynamic2DArray<double>(n_movies, lf);
        y = Dynamic2DArray<double>(n_movies, lf);

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
                y[i][j] = distribution(generator);
            }
        }
    }

    // u user, m movie
    double pred_one(int u, int m, vector<int> &r_u) {
        
        double sumY [lf] = {0};
        for (int j = 0; j < r_u.size(); j++) {
            double *y_r_u_j = y[r_u[j]];
            for (int i = 0; i < lf; i++) {
                sumY[i] += y_r_u_j[i];
            }
        }
        
        double dot = 0.0;
        double *p_u = p[u];
        double *q_m = q[m];
        double ru_negSqrt = 1.0 / pow(r_u.size(), 0.5);
        for (int i = 0; i < lf; i++){ // which latent factor
            dot += (p_u[i] + ru_negSqrt * sumY[i]) * q_m[i];
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
                
                vector<int> R_u = data.mpu[u];
                double ru_negSqrt = 1.0 / pow(R_u.size(), 0.5);
                
                double dr = pred_one(u, m, R_u) - data.val[idx[i]];
                
                b_u[u] -= lr * (dr + reg * b_u[u]);
                b_m[m] -= lr * (dr + reg * b_m[m]);
                
                double * p_u = p[u];
                double * q_m = q[m];
                double sumY [lf] = {0};
                for (int k = 0; k < R_u.size(); k++) {
                    double * y_r_u_k = y[R_u[k]];
                    for (int j = 0; j < lf; j++) {
                        sumY[j] += y_r_u_k[j];
                        y_r_u_k[j] -= lr * (dr * q_m[j] * ru_negSqrt + reg * y_r_u_k[j]);
                    }
                }
                
                for (int j = 0; j < lf; j++) {
                    double p_old = p_u[j];
                    double q_old = q_m[j];
                    p_u[j] -= lr * (dr * q_old + reg * p_old);
                    q_m[j] -= lr * (dr * (p_old + ru_negSqrt * sumY[j]) + reg * q_old);
                }
            }
        }
    }

    vector<double> predict(Dataset data) {
        vector<double> pred;
        for(int i = 0; i < data.row.size(); i++) {
            pred.push_back(pred_one(data.row[i], data.col[i], data.mpu[data.row[i]]));
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

class TimeSVDpp {
    public:
    double mean; // average of all ratings (constant)
    double *b_u; // user biases
    double *a_u; // user dev-time biases
    vector< unordered_map<int, double> > b_ut; // user time biases, initialize at train time
    double *b_m; // movie biases
    double **b_mt; // movie time-bin biases
    double **p; // user embeddings
    double **a_uk; // user dev-time embeddings
    vector< unordered_map<int, double*> > p_t; // user time embeddings, allocate and initialize at train time
    double **q; // movie embeddings
    double **y; // implicit latent factor
    double *u_tmean; // average user times (constant), initialize at train time
    
    int lf;

    TimeSVDpp(int latent_factors) {
        lf = latent_factors;
        b_u = new double[n_users];
        a_u = new double[n_users];
        b_m = new double[n_movies];
        b_mt = Dynamic2DArray<double>(n_movies, n_bins);
        p = Dynamic2DArray<double>(n_users, lf);
        a_uk = Dynamic2DArray<double>(n_users, lf);
        q = Dynamic2DArray<double>(n_movies, lf);
        y = Dynamic2DArray<double>(n_movies, lf);
        u_tmean = new double[n_users];

        default_random_engine generator;
        normal_distribution<double> distribution(0.0, 0.1);

        for (int i = 0; i < n_users; i++) {
            b_u[i] = distribution(generator);
            a_u[i] = distribution(generator);
            for (int j = 0; j < lf; j++) {
                p[i][j] = distribution(generator);
                a_uk[i][j] = distribution(generator);
            }
        }
        for (int i = 0; i < n_movies; i++) {
            b_m[i] = distribution(generator);
            for (int b = 0; b < n_bins; b++) {
                b_mt[i][b] = distribution(generator);
            }
            for (int j = 0; j < lf; j++) {
                q[i][j] = distribution(generator);
                y[i][j] = distribution(generator);
            }
        }
    }

    // u user, m movie
    double pred_one(int u, int m, vector<int> &r_u, int t, int bin_t) {
        
        double sumY [lf] = {0};
        for (int j = 0; j < r_u.size(); j++) {
            double *y_r_u_j = y[r_u[j]];
            for (int i = 0; i < lf; i++) {
                sumY[i] += y_r_u_j[i];
            }
        }
        
        double d_t = t - u_tmean[u];
        double dev_t = ((dt > 0) - (dt < 0)) * pow(fabs(dt), beta);
        double b_u_t = b_u[u] + a_u[u] * dev_t + b_ut[u][t];
        double b_m_t = b_m[m] + b_mt[m][bin_t];
        
        double *p_u = p[u];
        double *a_uk_u = a_uk[u];
        double *p_ut_u_t = p_ut[u][t];
        double *q_m = q[m];
        double dot = 0.0;
        double ru_negSqrt = 1.0 / pow(r_u.size(), 0.5);
        for (int i = 0; i < lf; i++){ // which latent factor
            double p_u_t = p_u[i] + a_uk_u[i] * dev_t + p_ut_u_t[i];
            dot += (p_u_t + ru_negSqrt * sumY[i]) * q_m[i];
        }
        return (dot + b_u_t + b_m_t + mean);
    }

    void train(Dataset &data, int epochs, double lr, double reg) {
        mean = accumulate(data.val.begin(), data.val.end(), 0.0) / data.row.size();
        vector<int> idx;
        for (int i = 0; i < data.row.size(); i++) {
            idx.push_back(i);
        }
        
        default_random_engine generator;
        normal_distribution<double> distribution(0.0, 0.1);
        for (int u = 0; u < n_users; u++) {
            u_tmean[u] = data.mean_date[u];
            for (int t: data.unique_dpu[u]) {
                b_ut[u][t] = distribution(generator);
                p_t[u][t] = new double[lf];
                for (int i = 0; i < lf; i++) {
                    p_t[u][t][i] = distribution(generator);
                }
            }
        }

        for (int ep = 0; ep < epochs; ep++) {
            cout << "Epoch " << ep << endl;
            random_shuffle(idx.begin(), idx.end());
            for (int i = 0; i < data.row.size(); i++) {
                int u = data.row[idx[i]];
                int m = data.col[idx[i]];
                
                vector<int> R_u = data.mpu[u];
                double ru_negSqrt = 1.0 / pow(R_u.size(), 0.5);
                
                double dr = pred_one(u, m, R_u) - data.val[idx[i]];
                
                b_u[u] -= lr * (dr + reg * b_u[u]);
                b_m[m] -= lr * (dr + reg * b_m[m]);
                
                double * p_u = p[u];
                double * q_m = q[m];
                double sumY [lf] = {0};
                for (int k = 0; k < R_u.size(); k++) {
                    double * y_r_u_k = y[R_u[k]];
                    for (int j = 0; j < lf; j++) {
                        sumY[j] += y_r_u_k[j];
                        y_r_u_k[j] -= lr * (dr * q_m[j] * ru_negSqrt + reg * y_r_u_k[j]);
                    }
                }
                
                for (int j = 0; j < lf; j++) {
                    double p_old = p_u[j];
                    double q_old = q_m[j];
                    p_u[j] -= lr * (dr * q_old + reg * p_old);
                    q_m[j] -= lr * (dr * (p_old + ru_negSqrt * sumY[j]) + reg * q_old);
                }
            }
        }
    }

    vector<double> predict(Dataset &data) {
        vector<double> pred;
        for(int i = 0; i < data.row.size(); i++) {
            pred.push_back(pred_one(data.row[i], data.col[i], data.mpu[data.row[i]]));
        }
        return pred;
    }

    double error(Dataset &data) {
        vector<double> pred = this->predict(data);
        double SE = 0.0;
        for(int i = 0; i < data.row.size(); i++) {
            SE += pow(pred[i] - data.val[i], 2);
        }
        return (pow(SE / data.row.size(), 0.5));
    }
};

//function to compute average
double compute_average(vector<int> &vi) {

  double sum = 0;

  // iterate over all elements
  for (int p:vi){
     sum = sum + p;
  }

  return (sum/vi.size());
 }

// r user, c movie, v rating
Dataset load_data(string path) {
    Dataset data;

    io::CSVReader<4> in(path);
    in.read_header(io::ignore_extra_column, "User Number", "Movie Number", "Date Number", "Rating", "bin");

    int r, c, t, v, b;
    while(in.read_row(r, c, t, v, b)) {
        data.row.push_back(r - 1);
        data.col.push_back(c - 1);
        data.val.push_back(v);
        data.day.push_back(t);
        data.bin.push_back(b);
        data.mpu[r-1].push_back(c - 1);
        data.dpu[r-1].push_back(t);
        data.unique_dpu[r-1].insert(t);
    }

    for(int i = 0; i < n_users; i++) {
        double avg = compute_average(data.dpu[i]);
        data.mean_date.push_back(avg);
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
    int latentFactors = 20;
    int epochs = 10;
    double lr = 0.005;
    double reg = 0.02;

    cout << "Loading data..." << endl;

    Dataset train_set = load_data("../data/mu_train.csv");
    Dataset test_set1 = load_data("../data/mu_probe.csv");
    Dataset test_set2 = load_data("../data/mu_qual.csv");

    cout << "Training model..." << endl;
    cout << "Using "<< latentFactors <<" latent factors "<<endl;
    cout << "Epochs: "<< epochs << " learning rate: "<< lr <<" reg: "<< reg << endl;

    clock_t t = clock();

    SVDpp model(latentFactors);
    cout<<"Model initalized! model used to train mu_probe.csv"<<endl;
    model.train(test_set1, epochs, lr, reg);

    double t_delta = (double) (clock() - t) / CLOCKS_PER_SEC;

    printf("Training time: %.2f s\n", t_delta);

    double rmse = model.error(test_set1);
    printf("Train RMSE: %.3f\n", rmse);
    rmse = model.error(test_set1);
    printf("Val RMSE: %.3f\n", rmse);
    
    /* UNCOMMENT TO SAVE SUBMISSION
    vector<double> predictions = model.predict(test_set1);
    save_submission("time_svd++", "mu", "probe", predictions);
    predictions = model.predict(test_set2);
    save_submission("time_svd++", "mu", "qual", predictions);
    */
}
