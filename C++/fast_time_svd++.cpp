#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include "csv.h"

/*
 * DONT FORGET: compile with "-Ofast -march=native"
 */

#define n_users 458293
#define n_movies 17770
#define n_bins 30
#define beta 0.4
#define min_date 1.0
#define max_date 2243.0

using namespace std;

struct Example {
    int m;
    int t;
    int r;
    int bin_t;
    // double dev_t;
};

struct UserData {
    vector<Example> entries;
    vector<int> R_u;
    unordered_set<int> dates;
    double mean_t;
};

struct Dataset {
    vector<UserData> user_data;
    double mean;
    int n_examples;
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
    // vector< unordered_map<int, double*> > p_t; // user time embeddings, allocate and initialize at train time
    double **q; // movie embeddings
    double **y; // implicit latent factor
    double *u_tmean; // average user times (constant), assign at train time
    
    double *sumY;  //
    double *gradY; // used to batch gradient updates of y for a given user
                   // based on https://github.com/linzebing/timeSVDplusplus
    
    double lr_b_u = 0.007;
    double lr_a_u = 2e-5;
    double lr_b_ut = 0.007;
    double lr_b_m = 0.007;
    double lr_b_mt = 0.007;
    double lr_lf = 0.02;
    double lr_a_uk = 2e-5;
    double reg_b_u = 0.005;
    double reg_a_u = 10.0;
    double reg_b_ut = 0.005;
    double reg_b_m = 0.005;
    double reg_b_mt = 0.005;
    double reg_lf = 0.03;
    double reg_a_uk = 10.0;
    
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
        
        sumY = new double[lf];
        gradY = new double[lf];

        default_random_engine generator;
        normal_distribution<double> distribution(0.0, 0.01);

        for (int i = 0; i < n_users; i++) {
            b_u[i] = 0.0;
            a_u[i] = 0.0;
            for (int j = 0; j < lf; j++) {
                p[i][j] = distribution(generator);
                a_uk[i][j] = 0.0;
            }
        }
        for (int i = 0; i < n_movies; i++) {
            b_m[i] = 0.0;
            for (int b = 0; b < n_bins; b++) {
                b_mt[i][b] = 0.0;
            }
            for (int j = 0; j < lf; j++) {
                q[i][j] = distribution(generator);
                y[i][j] = 0.0;
            }
        }
        
        vector<unordered_map<int, double>> temp(n_users, unordered_map<int, double>({}));
        b_ut = temp;
        /*
        vector<unordered_map<int, double*>> temp2(n_users, unordered_map<int, double*>({}));
        p_t = temp2;
        */
    }

    double pred_one(int &u, int &m, double &ru_negSqrt, int &t, int &bin_t, double &dev_t) {
        
        double b_u_t = b_u[u] + a_u[u] * dev_t + b_ut[u][t];
        double b_m_t = b_m[m] + b_mt[m][bin_t];
        
        double * p_u = p[u];
        double * q_m = q[m];
        double * a_uk_u = a_uk[u];
        // double * p_t_u_t = p_t[u][t];
        double dot = 0.0;
        for (int i = 0; i < lf; i++){ // which latent factor
            double p_u_t = p_u[i] + a_uk_u[i] * dev_t  /*+ p_t_u_t[i] */;
            dot += (p_u_t + ru_negSqrt * sumY[i]) * q_m[i];
        }
        return (dot + b_u_t + b_m_t + mean);
    }
    
    // u user, m movie
    double pred_one_test(int u, int m, vector<int> &r_u, int t, int bin_t) {
        
        for(int i = 0; i < lf; i++) {
            sumY[i] = 0.0;
        }
        for (int r_u_j: r_u) {
            double *y_r_u_j = y[r_u_j];
            for (int i = 0; i < lf; i++) {
                sumY[i] += y_r_u_j[i];
            }
        }
        
        double dt = t - u_tmean[u];
        double dev_t = ((dt > 0) - (dt < 0)) * pow(fabs(dt), beta);
        double b_u_t = b_u[u] + a_u[u] * dev_t + b_ut[u][t];
        double b_m_t = b_m[m] + b_mt[m][bin_t];
        
        double *p_u = p[u];
        double *a_uk_u = a_uk[u];
        double *q_m = q[m];
        double dot = 0.0;
        double ru_negSqrt = (r_u.size() > 0) ? (1.0 / sqrt(r_u.size())) : 0.0;
        
        double p_u_t;
        /*
        double *p_ut_u_t = p_t[u][t];
        if (p_t[u][t] == NULL) {
            for (int i = 0; i < lf; i++){ // which latent factor
                p_u_t = p_u[i] + a_uk_u[i] * dev_t;
                dot += (p_u_t + ru_negSqrt * sumY[i]) * q_m[i];
            }
        }
        else {
            for (int i = 0; i < lf; i++){ // which latent factor
                p_u_t = p_u[i] + a_uk_u[i] * dev_t + p_ut_u_t[i];
                dot += (p_u_t + ru_negSqrt * sumY[i]) * q_m[i];
            }
        }
        */
        // BEGIN temp block
        for (int i = 0; i < lf; i++){ // which latent factor
            p_u_t = p_u[i] + a_uk_u[i] * dev_t;
            dot += (p_u_t + ru_negSqrt * sumY[i]) * q_m[i];
        }
        // END temp block
        
        return (dot + b_u_t + b_m_t + mean);
    }

    void train(Dataset &data, Dataset &val_data, int epochs) {
        
        mean = data.mean;
        
        for (int u = 0; u < n_users; u++) {
            u_tmean[u] = data.user_data[u].mean_t;
            for (int t: data.user_data[u].dates) {
                b_ut[u][t] = 0.0;
                /*
                p_t[u][t] = new double[lf];
                for (int i = 0; i < lf; i++) {
                    p_t[u][t][i] = 0.0;
                }
                */
            }
        }
        
        vector<int> idx;
        for (int i = 0; i < n_users; i++) {
            idx.push_back(i);
        }
        
        for (int ep = 0; ep < epochs; ep++) {
            
            // cout << "Train RMSE: " << error(data, data) << endl;
            cout << "Val RMSE: " << error(val_data, data) << endl;
            cout << "Epoch " << ep << endl;
            random_shuffle(idx.begin(), idx.end());
            
            for (int u: idx) {
                vector<int> &R_u = data.user_data[u].R_u;
                double ru_negSqrt = (R_u.size() > 0) ? (1.0 / sqrt(R_u.size())) : 0.0;
                double mean_t = data.user_data[u].mean_t;
                
                for(int j = 0; j < lf; j++) {
                    sumY[j] = 0.0;
                    gradY[j] = 0.0;
                }
                for (int R_u_k: R_u) {
                    for (int j = 0; j < lf; j++) {
                        sumY[j] += y[R_u_k][j];
                    }
                }
                
                random_shuffle(data.user_data[u].entries.begin(), data.user_data[u].entries.end());
                
                for (Example ex: data.user_data[u].entries) {
                    int m = ex.m;
                    int t = ex.t;
                    int bin_t = ex.bin_t;
                    
                    double dt = t - mean_t;
                    double dev_t = ((dt > 0) - (dt < 0)) * pow(fabs(dt), beta);
                    
                    double dr = pred_one(u, m, ru_negSqrt, t, bin_t, dev_t) - ex.r;
                    
                    b_u[u] -= lr_b_u * (dr + reg_b_u * b_u[u]);
                    a_u[u] -= lr_a_u * (dr * dev_t + reg_a_u * a_u[u]);
                    b_ut[u][t] -= lr_b_ut * (dr + reg_b_ut * b_ut[u][t]);
                    
                    b_m[m] -= lr_b_m * (dr + reg_b_m * b_m[m]);
                    b_mt[m][bin_t] -= lr_b_mt * (dr + reg_b_mt * b_mt[m][bin_t]);
                    
                    double * q_m = q[m];
                    double * p_u = p[u];
                    double * a_uk_u = a_uk[u];
                    // double * p_ut_u_t = p_t[u][t];
                    
                    for (int j = 0; j < lf; j++) {
                        double p_old = p_u[j] + a_uk_u[j] * dev_t /*+ p_ut_u_t[j]*/;
                        double q_old = q_m[j];
                        gradY[j] += dr * q_m[j] * ru_negSqrt;
                        p_u[j] -= lr_lf * (dr * q_old + reg_lf * p_u[j]);
                        a_uk_u[j] -= lr_a_uk * (dr * q_old * dev_t + reg_a_uk * a_uk_u[j]);
                        // p_ut_u_t[j] -= (0.004) * decay * (dr * q_old + (0.01) * p_ut_u_t[j]);
                        q_m[j] -= lr_lf * (dr * (p_old + ru_negSqrt * sumY[j]) + reg_lf * q_m[j]);
                    }
                }
                
                for (int R_u_k: R_u) {
                    double * y_r_u_k = y[R_u_k];
                    for (int j = 0; j < lf; j++) {
                        y_r_u_k[j] -= lr_lf * (gradY[j] + reg_lf * y_r_u_k[j]);
                    }
                }
            }
            
            lr_b_u *= 0.9;
            lr_a_u *= 0.9;
            lr_b_ut *= 0.9;
            lr_b_m *= 0.9;
            lr_b_mt *= 0.9;
            lr_lf *= 0.9;
            lr_a_uk *= 0.9;
        }
        
    }

    vector<double> predict(Dataset &data, Dataset &train_data) {
        vector<double> pred;
        for(int u = 0; u < n_users; u++) {
            vector<int> R_u = train_data.user_data[u].R_u;
            for(Example ex: data.user_data[u].entries) {
                pred.push_back(pred_one_test(u, ex.m, R_u, ex.t, ex.bin_t));
            }
        }
        return pred;
    }

    double error(Dataset &data, Dataset &train_data) {
        vector<double> pred = predict(data, train_data);
        vector<double> trut;
        for(int u = 0; u < n_users; u++) {
            for(Example ex: data.user_data[u].entries) {
                trut.push_back(ex.r);
            }
        }
        double SE = 0.0;
        for(int i = 0; i < pred.size(); i++) {
            SE += pow(pred[i] - trut[i], 2);
        }
        return (pow(SE / pred.size(), 0.5));
    }
};

//function to compute average
double compute_average(vector<int> &vi) {
    
    double sum = 0;

    // iterate over all elements
    for (int p:vi){
     sum = sum + p;
    }

    if (vi.size() > 0) {
      return (sum/vi.size());
    }
    else {
      return 0.0;
    }
}

// r user, c movie, v rating
Dataset load_data(string path) {
    Dataset data;
    data.user_data.resize(n_users);

    io::CSVReader<4> in(path);
    in.read_header(io::ignore_extra_column, "User Number", "Movie Number", "Date Number", "Rating");
    
    double sum = 0.0;
    data.n_examples = 0;
    double bin_size = (1 + max_date - min_date) / n_bins;
    int u, m, t, r, bin_t;
    while(in.read_row(u, m, t, r)) {
        bin_t = (t - min_date) / bin_size;
        Example ex = {m-1, t, r, bin_t};
        data.user_data[u-1].entries.push_back(ex);
        data.user_data[u-1].dates.insert(t);
        data.user_data[u-1].R_u.push_back(m-1);
        sum += r;
        data.n_examples++;
    }
    data.mean = sum / data.n_examples;
    for(int u = 0; u < n_users; u++) {
        vector<int> times;
        for (Example ex: data.user_data[u].entries) {
            times.push_back(ex.t);
        }
        double avg = compute_average(times);
        data.user_data[u].mean_t = avg;
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
    int latentFactors = 100;
    int epochs = 30;
    double lr = 0.02;  //
    double reg = 0.03; // these parameters are currently hard-coded

    cout << "Loading data..." << endl;

    Dataset train_set = load_data("../data/real_mu_train.csv");
    Dataset test_set1 = load_data("../data/real_um_probe_sorted.csv");
    Dataset test_set2 = load_data("../data/um_qual.csv");

    cout << "Training model..." << endl;
    cout << "Using "<< latentFactors <<" latent factors "<<endl;
    cout << "Epochs: "<< epochs << " learning rate: "<< lr <<" reg: "<< reg << endl;

    clock_t t = clock();

    TimeSVDpp model(latentFactors);
    cout<<"Model initialized! model used to train real_mu_train.csv"<<endl;
    model.train(train_set, test_set1, epochs);

    double t_delta = (double) (clock() - t) / CLOCKS_PER_SEC;

    printf("Training time: %.2f s\n", t_delta);

    // double rmse = model.error(train_set, train_set);
    // printf("Train RMSE: %.3f\n", rmse);
    double rmse = model.error(test_set1, train_set);
    printf("Val RMSE: %.3f\n", rmse);
    
    vector<double> predictions = model.predict(test_set1, train_set);
    save_submission("time_svd++", "um", "real_probe", predictions);
    predictions = model.predict(test_set2, train_set);
    save_submission("time_svd++", "um", "qual", predictions);

}
