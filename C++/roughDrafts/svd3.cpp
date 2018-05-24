#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <numeric>
#include <random>
#include <algorithm>
#include "csv.h"
#include <thread>

#define n_users 458293
#define n_movies 17770

using namespace std;


class Dataset {
    public:
    vector<int> user_ids; // user id
    vector<int> movie_ids; // movie id
    vector<int> val; // rating
    // movies per user what movies each user has rated [user_ids][movie_ids]
    vector<vector<int>> mpu;  
    Dataset() {
        vector<vector<int>> temp(n_users, vector<int>(0,0));
        mpu = temp;
    }
};

Dataset load_data(string path); 
void save_submission(string model_name, string ordering, string source, vector<double> predictions); 
int updateSVDpp(int u, int m, double dr, int start, int end, double lr, double reg, double **p, double **q, double **y, vector<vector<int>> &mpu); 

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
    
    double train(Dataset data, int epochs, double lr, double reg) {
        mean = accumulate(data.val.begin(), data.val.end(), 0.0) / data.user_ids.size();
        vector<int> idx;
        for (int i = 0; i < data.user_ids.size(); i++) {
            idx.push_back(i);
        }
        
        for (int ep = 0; ep < epochs; ep++) {
            cout << "Epoch " << ep << endl;
            random_shuffle(idx.begin(), idx.end());
            for (int i = 0; i < data.user_ids.size(); i++) {
                int u = data.user_ids[idx[i]];
                int m = data.movie_ids[idx[i]];
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
        for(int i = 0; i < data.user_ids.size(); i++) {
            pred.push_back(pred_one(data.user_ids[i], data.movie_ids[i]));
        }
        return pred;
    }
    
    double error(Dataset data) {
        vector<double> pred = this->predict(data);
        double SE = 0.0;
        for(int i = 0; i < data.user_ids.size(); i++) {
            SE += pow(pred[i] - data.val[i], 2);
        }
        return (pow(SE / data.user_ids.size(), 0.5));
    }
};

class SVDpp {
    public:
    int numThreads; 
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
    double pred_one(int u, int m, vector<int> r_u) {
        double dot = 0.0;
        double sumY [lf] = {0};
        double ru_negSqrt = 1.0 / pow(r_u.size(), 2.0);

        for (int i = 0; i < lf; i++){ // which latent factor
            for (int j = 0; j < r_u.size(); j++) { // movie watched, for user u
                sumY[i] += y[r_u[j]][i];
            }
            dot += (p[u][i] + ru_negSqrt * sumY[i]) * q[m][i];
        }
        return (dot + b_u[u] + b_m[m] + mean);
    }

    double train(Dataset data, int epochs, double lr, double reg) {
        mean = accumulate(data.val.begin(), data.val.end(), 0.0) / data.user_ids.size();
        vector<int> idx;
        for (int i = 0; i < data.user_ids.size(); i++) {
            idx.push_back(i);
        }

        for (int ep = 0; ep < epochs; ep++) {
            cout << "Epoch " << ep << endl;
            random_shuffle(idx.begin(), idx.end());
            for (int i = 0; i < data.user_ids.size(); i++) {
                int u = data.user_ids[idx[i]];
                int m = data.movie_ids[idx[i]];
                double dr = pred_one(u, m, data.mpu[u]) - data.val[idx[i]];
                b_u[u] -= lr * (dr + reg * b_u[u]);
                b_m[m] -= lr * (dr + reg * b_m[m]);

                // NOTE: THE NUMBER OF THREADS ARE HARD CODED =====================================
                // Can put into a vector if you want though
                // know that we have 100 latent factors, so made a thread for every 10
                // note that ref() is a reference wrapper REQUIRED FOR THREADING, DO NOT REMOVE
                // http://www.cplusplus.com/reference/functional/reference_wrapper/
                // https://stackoverflow.com/questions/28950835/c-error-no-type-named-type-in-class-stdresult-ofvoid-stdunordered
    
                thread th1(updateSVDpp, u, m, dr, 0, 20, lr, reg, p, q, y, ref(data.mpu));
                thread th2(updateSVDpp, u, m, dr, 20, 40, lr, reg, p, q, y, ref(data.mpu));
                thread th3(updateSVDpp, u, m, dr, 40, 60, lr, reg, p, q, y, ref(data.mpu));
                thread th4(updateSVDpp, u, m, dr, 60, 80, lr, reg, p, q, y, ref(data.mpu));
                thread th5(updateSVDpp, u, m, dr, 80, 100, lr, reg, p, q, y, ref(data.mpu));
                    
                th1.join();
                th2.join();
                th3.join();
                th4.join();
                th5.join();
            }
        }
    }
    
    vector<double> predict(Dataset data) {
        vector<double> pred;
        for(int i = 0; i < data.user_ids.size(); i++) {
            pred.push_back(pred_one(data.user_ids[i], data.movie_ids[i], data.mpu[data.user_ids[i]]));
        }
        return pred;
    }
    
    double error(Dataset data) {
        vector<double> pred = this->predict(data);
        double SE = 0.0;
        for(int i = 0; i < data.user_ids.size(); i++) {
            SE += pow(pred[i] - data.val[i], 2);
        }
        return (pow(SE / data.user_ids.size(), 0.5));
    }
};

// this is a static function so that we can make this from a thread
// made it an integer return because we're calling this for the thread
int updateSVDpp(int u, int m, double dr, int start, int end, double lr, double reg, double **p, double **q, double **y, vector<vector<int>> &mpu) { 
    //cout<<"update called with latent factor: "<<latFact<<endl;
    for (int latFact = start; latFact < end; latFact++) {
        double p_old = p[u][latFact];
        double q_old = q[m][latFact];
        double ru_negSqrt = 1.0 / pow(mpu[u].size(), 2.0);
                        
        // the summation of the y's for all the movies a user 
        // has watched with respect to latent factor latFact
        double sumY_mu = 0.0;
        for (int movInd = 0; movInd < mpu[u].size(); movInd++) {
            sumY_mu += y[mpu[u][movInd]][latFact];
        }
    
        p[u][latFact] -= lr * (dr * q_old + reg * p_old);
        q[m][latFact] -= lr * (dr * (p_old + ru_negSqrt * sumY_mu) + reg * q_old);
    
        // update y's here
        for (int movInd = 0; movInd < mpu[u].size(); movInd++) {
            double y_old = y[mpu[u][movInd]][latFact]; 
            y[mpu[u][movInd]][latFact] -= lr * (dr * q_old * ru_negSqrt + reg * y_old);
        }
    }
    return 1;
}

// r user, c movie, v rating
Dataset load_data(string path) {
    Dataset data;

    io::CSVReader<3> in(path);
    in.read_header(io::ignore_extra_column, "User Number", "Movie Number", "Rating");

    int r, c, v;
    while(in.read_row(r, c, v)) {
        data.user_ids.push_back(r - 1);
        data.movie_ids.push_back(c - 1);
        data.val.push_back(v);
        data.mpu[r-1].push_back(c - 1);
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

// note that we have to add the -pthread to our g++ compiler, so that we can run threads
// stackoverflow.com/questions/17264984/undefined-reference-to-pthread-create
int main(int argc, char *argv[]) {
    
    cout << "Loading data..." << endl;
    
    Dataset train_set = load_data("../data/mu_train.csv");
    Dataset test_set1 = load_data("../data/mu_probe.csv");
    Dataset test_set2 = load_data("../data/mu_qual.csv");
    
    cout << "Training model..." << endl;
    
    clock_t t = clock();
    
    SVDpp model(100);
    cout<<"Model initalized! training on mu_probe.csv"<<endl;
    model.train(test_set1, 10, 0.005, 0.02);
    
    double t_delta = (double) (clock() - t) / CLOCKS_PER_SEC;
    
    printf("Training time: %.2f s\n", t_delta);
    
    double rmse = model.error(test_set1);
    printf("Train RMSE: %.3f\n", rmse);
    rmse = model.error(test_set1);
    printf("Val RMSE: %.3f\n", rmse);
    
    vector<double> predictions = model.predict(test_set1);
    save_submission("svd++", "mu", "probe", predictions);
    predictions = model.predict(test_set2);
    save_submission("svd", "mu", "qual", predictions);
}
