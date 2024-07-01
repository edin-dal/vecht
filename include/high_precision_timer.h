#include <unordered_map>
#include <vector>
#include <math.h>
#include <numeric>
#include <chrono>

using namespace std::chrono;

using namespace std;

class HighPrecisionTimer {
private:
    int64_t StartTime;

public:
    unordered_map<int, vector<double>> times;

    HighPrecisionTimer() {
        StartTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    }

    void Reset() {
        StartTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    }

    void Clear() {
        times.clear();
    }

    long GetElapsedTime() {
        return (duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - StartTime);
    }

    void PrintElapsedTimeAndReset(string Title) {
        const auto& elapsedTime = GetElapsedTime();
        cout << ">> " << elapsedTime << "\t ms |\t" << Title << endl; 
        Reset();
    }

    long StoreElapsedTime(int qNo){
        const auto& elapsedTime = GetElapsedTime();
        times[qNo].push_back(elapsedTime);
        return elapsedTime;
    }

    double GetMean(int qNo)
    {
        auto& roundsVec = times[qNo];
        roundsVec.erase(roundsVec.begin());
        return  accumulate(roundsVec.begin(), roundsVec.end(), 0)/(roundsVec.size()+0.0);
    }


    double GetStDev(int qNo)
    {
        auto& roundsVec = times[qNo];
        roundsVec.erase(roundsVec.begin());
        auto mean = accumulate(roundsVec.begin(), roundsVec.end(), 0)/(roundsVec.size()+0.0);
        double sq_sum = inner_product(roundsVec.begin(), roundsVec.end(), roundsVec.begin(), 0.0);
        return std::sqrt(sq_sum / roundsVec.size() - mean * mean);
    }

    double GetTotal(int qNo)
    {
        auto& roundsVec = times[qNo];
        return accumulate(roundsVec.begin(), roundsVec.end(), 0);
    }

};