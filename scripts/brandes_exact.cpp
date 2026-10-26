// Exact unweighted Brandes betweenness, matching networkx normalised=True,
// endpoints=False, for an undirected graph. Input: n m \n m lines `u v`.
// Output: one betweenness value per vertex (0-based order).
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
using namespace std;

int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n,m;
    cin >> n >> m;
    vector<vector<int>> adj(n);

    for(int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        if(u == v) continue;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    for(auto& nbrs:adj)
        sort(nbrs.begin(), nbrs.end());

    int T = (argc > 1) ? atoi(argv[1]):4;
    T = max(1, min(T, n));
    vector<vector<double>> threadB(T, vector<double>(n, 0.));
    vector<thread> workers;

    for(int tid = 0; tid < T; tid++) workers.emplace_back([&,tid]() {
        vector<int> dist(n, -1), head(n, -1), order(n), predV(2 * m + 1), nextPred(2 * m + 1);
        vector<double> sigma(n), delta(n);
        auto& bet = threadB[tid];

        for(int s = tid; s < n; s += T){
            fill(dist.begin(), dist.end(), -1);
            fill(head.begin(), head.end(), -1);
            fill(sigma.begin(), sigma.end(), 0.);
            fill(delta.begin(), delta.end(), 0.);

            int first = 0, last = 0, edgesUsed = 0;
            dist[s] = 0;
            sigma[s] = 1.;
            order[last++] = s;

            while(first < last){
                int v = order[first++];
                int want = dist[v] + 1;
                for(int w:adj[v]){
                    if(dist[w] < 0){
                        dist[w] = want;
                        order[last++] = w;
                    }
                    if(dist[w] == want){
                        sigma[w] += sigma[v];
                        predV[edgesUsed] = v;
                        nextPred[edgesUsed] = head[w];
                        head[w] = edgesUsed++;
                    }
                }
            }
            for(int i = last - 1; i >= 0; --i){
                int w = order[i];
                for(int e = head[w]; e != -1; e = nextPred[e]){
                    int v = predV[e];
                    delta[v] += (sigma[v] / sigma[w]) * (1. + delta[w]);
                }
                if(w != s) bet[w] += delta[w];
            }
        }
    });

    for(auto& th:workers) th.join();
    cout << setprecision(17);
    double scale = n > 2 ? 1. / double(n - 1) / double(n - 2):0.;
    for(int v = 0; v < n; v++){
        double b = 0.;
        for(auto& part:threadB) b += part[v];
        cout << b * scale << '\n';
    }
}