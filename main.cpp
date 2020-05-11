#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <set>

#include <thread>

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cfloat>
/* 
*** STO NEVALJA ***

1. za heart-v dataset CART gini i HD imaju AUC manji od 0.5 lol

2. dosta toga je c/p, moglo bi se nekako generalizirati da CART i C4.5 budu bazne podjele,
    a racunanje split gain-a da bude neovisno o tome

3. seemingly Information Gain Ratio konzistentno polucuje losijim rezultatima od Information Gain
    - ?? nema smisli

*/

// TODO: dodaj grid search za hiperparametre | medium | low
// TODO: sve osim stats u log file or sth | ultra ez | low
// TODO: dodaj mogucnost spremanja stvorenog stabla | kinda ez-medium | low
// TODO: fixati apsolutno sve da nije ovako fugly | tricky | medium

// lol
// TODO: gpu????? | poprilicno tricky | ultra low
// TODO: hellinger net????? lol | ultra tricky delaj u pajtonima | hopefully nonexistent

// fpga support when

#define DEBUG(text) fprintf(stderr, "%d: %s\n", __LINE__, text);

//#define PROFILING
#ifdef PROFILING
#include "profiler.hpp"
#define BEGIN_SESSION(name) Instrumentor::get().beginSession(name);
#define END_SESSION() Instrumentor::get().endSession();
#define PROFILE_SCOPE(name) Timer timer##__LINE__(name);
#define PROFILE_FUNCTION PROFILE_SCOPE(__FUNCTION__)
#else
#define BEGIN_SESSION(name)
#define END_SESSION()
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION
#endif

const float MISSING = FLT_MAX;

#include <utility>
using Row = std::pair<std::vector<float>, float>;

//using Row = std::pair<std::vector<float>, float>;

//using Row = std::vector<float>;
//using Row = RowT<float>;
using Rows = std::vector<Row>;

std::vector<std::string> classes; // TODO: get rid of these globals, legitimately just send them to everything that uses them
std::vector<std::string> attrNames;
std::vector<std::vector<std::string>> attrValues; // att[0] == {"b", "o", "x"}
std::vector<bool> isContinuous;
std::vector<float> mins, maxs;

namespace Utils {
    const float epsilon = 1e-6;
    bool eq(float x, float y) {
        return fabs(x-y) <= epsilon;
    }

    Rows filter(const Rows &rows, int label) {
        // filter only for rows with label == `label`
        Rows new_rows;
        for (const auto &row: rows) {
            if ((int)row.first.back() == label) {
                new_rows.push_back(row);
            }
        }
        return new_rows;
    }

    int count(const Rows &rows, int label) {
        // count only for rows with label == `label`
        int ret = 0;
        for (const auto &row: rows) {
            if ((int)row.first.back() == label) {
                ++ret;
            }
        }
        return ret;
    }

    inline float sqr(float x) {
        return x*x;
    }

    void trim(std::string &s) {
        int start = 0, end = s.size()-1;
        while (s[start] == ' ') ++start;
        while (s[end] == ' ') --end;
        // "   abcdefgh   "
        //     ^      ^
        s = s.substr(start, end-start+1);
    }

    template<typename T>
    T accumulate(std::vector<T> v, T init = 0) {
        for (const auto &e: v) {
            init += e;
        }
        return init;
    }

    [[deprecated]] // unused but maybe it finds a purpose
    float mFast_Log2(float val) {
        union { float val; int32_t x; } u = { val };
        float log_2 = (float)(((u.x >> 23) & 255) - 128);              
        u.x   &= ~(255 << 23);
        u.x   += 127 << 23;
        log_2 += ((-0.34484843f) * u.val + 2.02466578f) * u.val - 0.67487759f; 
        return (log_2);
    } 

    float log(float x) {
        if (eq(x, 0)) {
            return 0;
        } else {
            return log2f(x);
        }
    }
}

struct Question {
    int column;
    float value;

    int match(const Row &example, bool c45 = false) const { // a0: 5, a1: "burek", a2: 6
        // Compare the feature value in an example to the
        // feature value in this question.
        if (example.first[column] == MISSING) {
            if (isContinuous[column]) {
                return 2;
            } else {
                return attrValues[column].size();
            }
        }
        if (isContinuous[column]) {
            return example.first[column] > value || Utils::eq(example.first[column], value);
        }
        if (c45) { // returns the index of the value in attrValues[column]
            return (int)example.first[column];
        }
        return (int)example.first[column] == (int)value;
    }
};

std::vector<float> class_histogram(const Rows &rows) {
    /*
    count number of rows per class
    */
    std::vector<float> histogram(classes.size(), 0);
    for (const auto &row: rows) {
        int label = row.first.back();
        histogram[label] += row.second;
    }
    return histogram;
}

struct Node {
    bool isLeaf = false;
    std::vector<Node *> children;
    Question question;
    std::vector<float> predictions;
    float total_weight = 0.0f;
};

Node *DecisionNodeContinuous(Node *l, Node *r, const Question &q) {
    Node *node = new Node();
    node->question = q;
    node->children = {l, r};
    return node;
}

Node *Leaf(const Rows &rows) {
    Node *node = new Node();
    node->isLeaf = true;
    node->predictions = class_histogram(rows);
    for (float e: node->predictions) {
        node->total_weight += e;
    }
    return node;
}

float gini(const Rows &rows) {
    /*
    Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    */
    auto histogram = class_histogram(rows);
    float impurity = 1;
    for (int label: histogram) {
        float probability = 1. * label / rows.size();
        impurity -= probability*probability;
    }
    return impurity;
}

float new_gini(const std::vector<int> &histogram) {
    float impurity = 1;
    for (int i = 0; i < (int)histogram.size()-1; ++i) { // -1 jer je zadnji broj redova
        int label = histogram[i];
        float probability = 1. * label / histogram.back();
        impurity -= probability*probability;
    }
    return impurity;
}

float entropy(const Rows &rows) {
    if (rows.size() == 0) return 0;
    auto num_classes = class_histogram(rows);
    float ent = 0;
    for (int num: num_classes) {
        float x = 1.*num/rows.size();
        ent += x*Utils::log(x);
    }
    return -ent;
}

float new_entropy(const std::vector<float> &num_classes) {
    if (num_classes.back() == 0) return 0;
    float ent = 0;
    for (int i = 0; i < (int)num_classes.size()-1; ++i) { // -1 jer je zadnji broj redova
        int num = num_classes[i];
        float x = 1.*num/num_classes.back();
        ent += x*Utils::log(x);
    }
    return -ent;
}

float informationGain(const Rows &rows, const std::vector<Rows> &splits, float beforeEntropy) {
    float afterEntropy = 0;
    for (int i = 0; i < (int)splits.size(); ++i) {
        float weight = 1. * splits[i].size() / rows.size();
        afterEntropy += weight * entropy(splits[i]);
    }
    return beforeEntropy - afterEntropy;
}

float new_informationGain(int rowSize, const std::vector<std::vector<float>> &splits, float beforeEntropy) {
    float afterEntropy = 0;
    for (int i = 0; i < (int)splits.size()-1; ++i) {
        float weight = 1. * splits[i].back() / rowSize;
        afterEntropy += weight * new_entropy(splits[i]);
    }
    return beforeEntropy - afterEntropy;
}

float intrinsicValue(const Rows &rows, const std::vector<Rows> &splits) {
    float iv = 0;
    for (const auto& split: splits) {
        float weight = 1. * split.size() / rows.size();
        iv += weight * Utils::log(weight);
    }
    return -iv;
}

float informationGainRatio(const Rows &rows, const std::vector<Rows> &splits, float beforeEntropy) {
    return informationGain(rows, splits, beforeEntropy) / intrinsicValue(rows, splits);
}

float new_informationGainRatio(int rowSize, const std::vector<std::vector<float>> &splits, float beforeEntropy) {
    float afterEntropy = 0;
    float iv = 0;
    for (int i = 0; i < (int)splits.size()-1; ++i) {
        float weight = 1. * splits[i].back() / rowSize;
        afterEntropy += weight * new_entropy(splits[i]);
        iv += weight * Utils::log(weight);
    }
    return (beforeEntropy - afterEntropy) / -iv;
}

std::set<float> feature_values(int col, const Rows &rows) {
    // TODO: ne moze ostati float, ovisit ce o tipu attr
    // *mozda* nas to nije briga?
    std::set<float> values;
    for (const auto &row: rows) {
        values.insert(row.first[col]);
    }
    return values;
}

template<typename T>
void partition_class_histogram(const Rows &rows, const Question &question, bool c45, std::vector<std::vector<T>> &splits) {
    for (const auto &row: rows) {
        splits[question.match(row, c45)][(int)row.first.back()] += row.second;
        splits[question.match(row, c45)].back() += row.second;
    }
}

template<int>
void partition_class_histogram(const Rows &rows, const Question &question, bool c45, std::vector<std::vector<int>> &splits) {
    for (const auto &row: rows) {
        ++splits[question.match(row, c45)][(int)row.first.back()];
        ++splits[question.match(row, c45)].back();
    }
}

// partition continuous
Rows partition(const Rows &rows, const Question &question, bool c45, Rows &true_rows, Rows &false_rows, float &total_weight) {
    Rows missing;
    for (const auto &row: rows) {
        total_weight += row.second;
        int match = question.match(row, c45);
        if (match == 2) {
            missing.push_back(row);
        } else if (match == 1) {
            true_rows.push_back(row);
        } else {
            false_rows.push_back(row);
        }
    }
    return missing;
}

// partition discrete
Rows partition(const Rows &rows, const Question &question, bool c45, std::vector<Rows> &splits, float &total_weight) {
    Rows missing;
    splits.resize(attrValues[question.column].size());
    for (const auto &row: rows) {
        total_weight += row.second;
        int match = question.match(row, c45);
        if (match == (int)attrValues[question.column].size()) {
            missing.push_back(row);
        } else {
            splits[match].push_back(row);
        }
    }
    return missing;
}

/* <google> */
float gini_gain(const Rows &left, const Rows &right,
                float current_uncertainty) {
    float p = 1. * left.size() / (left.size() + right.size());    
    return current_uncertainty - p*gini(left) - (1-p)*gini(right);
}

float new_gini_gain(const std::vector<std::vector<int>> &splits,
                    float current_uncertainty) {
    float p = 1. * splits[0].back() / (splits[0].back() + splits[1].back());
    return current_uncertainty - p*new_gini(splits[0]) - (1-p)*new_gini(splits[1]);
}

void split_continuous(const Rows &rows, int column, float current_uncertainty,
                      float &best_gain, Question &best_question,
                      int max_buckets, bool c45) {
    std::set<float> values;
    if (max_buckets != 0) {
        float min = mins[column], max = maxs[column];
        float step = (max-min)/max_buckets;
        for (int bucket = 0; bucket <= max_buckets; ++bucket) {
            values.insert(bucket*step+min);
        }
    } else {
        values = feature_values(column, rows);
    }
    int size = 2+1; // +1 za missing
    std::vector<std::vector<int>> splits(size, std::vector<int>(classes.size()+1, 0)); // +1 za broj redova u tom splitu
    for (float value: values) {
        Question question = {column, value};
        for (auto &split: splits) for (int &e: split) e = 0.0;
        partition_class_histogram(rows, question, c45, splits);
        if (splits[0].back() == 0 || splits[1].back() == 0) continue;
        float gain = new_gini_gain(splits, current_uncertainty);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void split_discrete(const Rows &rows, int column, float current_uncertainty,
                    float &best_gain, Question &best_question,
                    bool c45) {
    int size = attrValues[column].size()+1; // +1 za missing
    std::vector<std::vector<int>> splits(size, std::vector<int>(classes.size()+1, 0)); // +1 za broj redova u tom splitu
    for (int value = 0; value < (int)attrValues[column].size(); ++value) {
        Question question = {column, (float)value};
        Rows true_rows, false_rows;
        for (auto &split: splits) for (int &e: split) e = 0;
        partition_class_histogram(rows, question, c45, splits);
        if (splits[0].back() == 0 || splits[1].back() == 0) continue;
        float gain = new_gini_gain(splits, current_uncertainty);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void find_best_split_google(Rows &rows, float &best_gain, 
                            Question &best_question, int max_buckets, bool c45) {
    best_gain = 0;
    float current_uncertainty = gini(rows);
    int n_features = rows[0].first.size() - 1;
    for (int column = 0; column < n_features; ++column) {
        if (isContinuous[column]) {
            split_continuous(rows, column, current_uncertainty, best_gain, best_question, max_buckets, c45);
        } else {
            split_discrete(rows, column, current_uncertainty, best_gain, best_question, c45);
        }
    }
}
/* </google> */

/* <hellinger> */
void hellinger_split_continuous(Rows &rows, int column,
                                float &best_gain, Question &best_question,
                                int) {
    std::sort(rows.begin(), rows.end(), [column](const Row &a, const Row &b){
        return a.first[column] < b.first[column];
    });
    std::vector<int> leftFreq(classes.size()), rightFreq(classes.size());
    int S = 0, toSub = 0;
    for (const auto &row: rows) {
        if (Utils::eq(row.first[column], MISSING)) {
            ++toSub;
        } else {
            ++rightFreq[(int)row.first.back()];
            ++S;
        }
    }
    if (S < 2) {
        // not enough values to branch, do not pass go, do not collect howevermany dollars
        return;
    }
    int rsize = rows.size(), lsize = 0;
    for (int i = 1; i < (int)rows.size()-1; ++i) {
        if (Utils::eq(rows[i].first[column], MISSING)) break;
        --rightFreq[(int)rows[i].first.back()];
        --rsize;
        ++leftFreq[(int)rows[i-1].first.back()];
        ++lsize;
        if (rsize == 0) break;
        if (Utils::eq(rows[i].first[column], rows[i-1].first[column])) continue;
        float sum = 0;
        int pairs = classes.size()*(classes.size()-1)/2;
        for (int c1 = 0; c1 < (int)classes.size()-1; ++c1) {
            for (int c2 = c1+1; c2 < (int)classes.size(); ++c2) {
                //find size of X1 and X2
                float nX1 = rightFreq[c1] + leftFreq[c1];
                float nX2 = rightFreq[c2] + leftFreq[c2];
                //since attribute values can only be "left" or "right" only two terms in summation
                float radicand = Utils::sqr(sqrt(1.*rightFreq[c2]/nX2) - sqrt(1.*rightFreq[c1]/nX1))+
                                 Utils::sqr(sqrt(1.*leftFreq[c2]/nX2) - sqrt(1.*leftFreq[c1]/nX1));
                sum += sqrt(radicand);
                //DEBUG("BRUH");
            }
        }
        float gain = 1.*sum/pairs;
        //printf("%f/%d = %f\n", sum, pairs, gain);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = {column, rows[i].first[column]};
        }
    }
}

void hellinger_split_discrete(const Rows &rows, int column,
                              float &best_gain, Question &best_question) {
    std::vector<std::vector<int>> attrClassCnts(attrValues[column].size(), std::vector<int>(classes.size()));
    Rows missing;
    std::vector<int> valueCounts(attrValues[column].size());
    for (const auto &row: rows) {
        if (Utils::eq(row.first[column], MISSING)) {
            missing.push_back(row);
            continue;
        }
        ++attrClassCnts[(int)row.first[column]][(int)row.first.back()]; // I'm not sure if CART uses weighted rows?
        ++valueCounts[(int)row.first[column]];
    }
    // find the value that contains the most instances/rows
    int maxValueCount = std::max_element(valueCounts.begin(), valueCounts.end())-valueCounts.begin();
    valueCounts[maxValueCount] += missing.size();
    // place the missing instances into that one
    for (const auto &row: missing) {
        ++attrClassCnts[maxValueCount][(int)row.first.back()];
    }
    int pairs = classes.size()*(classes.size()-1)/2;
    for (int value = 0; value < (int)attrValues[column].size(); ++value) {
        Question question = {column, (float)value};
        float sum = 0.0f;
        int rsize = valueCounts[value];
        int lsize = rows.size() - rsize;
        if (lsize == 0 || rsize == 0) continue;
        for (int c1 = 0; c1 < (int)classes.size()-1; ++c1) {
            for (int c2 = c1+1; c2 < (int)classes.size(); ++c2) {
                float excludeCount[2] = {};
                float includeCount[2] = {(float)attrClassCnts[value][c1], (float)attrClassCnts[value][c2]};
                for (int exValue = 0; exValue < (int)attrValues[column].size(); ++exValue) {
                    if (value == exValue) continue;
                    excludeCount[0] += attrClassCnts[exValue][c1];
                    excludeCount[1] += attrClassCnts[exValue][c2];
                }
                float nX1 = includeCount[0] + excludeCount[0];
                float nX2 = includeCount[1] + excludeCount[1];

                float radicand = Utils::sqr(sqrt(1.*includeCount[1]/nX2) - sqrt(1.*includeCount[0]/nX1))+
                                 Utils::sqr(sqrt(1.*excludeCount[1]/nX2) - sqrt(1.*excludeCount[0]/nX1));
                sum += sqrt(radicand);
            }
        }

        float gain = sum/pairs;
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void find_best_split_hellinger(Rows &rows,
                               float &best_gain, Question &best_question,
                               int max_buckets, bool) {
    best_gain = 0;
    int n_features = rows[0].first.size() - 1;
    for (int column = 0; column < n_features; ++column) {
        if (isContinuous[column]) {
            hellinger_split_continuous(rows, column, best_gain, best_question, max_buckets);
        } else {
            hellinger_split_discrete(rows, column, best_gain, best_question);
        }
    }
}

void new_hellinger_split_continuous(Rows &rows, int column,
                                    float &best_gain, Question &best_question) {
    std::sort(rows.begin(), rows.end(), [column](const Row &a, const Row &b){
        return a.first[column] < b.first[column];
    });
    std::vector<float> leftFreq(classes.size()), rightFreq(classes.size());
    float S = 0.0f, toSub = 0.0f;
    for (const auto &row: rows) {
        if (Utils::eq(row.first[column], MISSING)) {
            toSub += row.second;
        } else {
            rightFreq[(int)row.first.back()] += row.second;
            S += row.second;
        }
    }
    float minLeaf = std::min(25.0f, std::max(0.1f*S/(classes.size()), 2.0f)); // 2 should prolly be a variable oh well
    if (S < 2*minLeaf) {
        // not enough values to branch, do not pass go, do not collect howevermany dollars
        return;
    }
    int rsize = rows.size(), lsize = 0;
    for (int i = 1; i < (int)rows.size()-1; ++i) {
        const auto &row = rows[i];
        if (row.first[column] == MISSING) break;
        float value = rows[i-1].first[column];
        rightFreq[(int)row.first.back()] -= row.second;
        if (Utils::eq(rightFreq[(int)row.first.back()], 0)) {
            rightFreq[(int)row.first.back()] = 0.0f;
        }
        --rsize;
        leftFreq[(int)rows[i-1].first.back()] += rows[i-1].second;
        ++lsize;
        if (lsize == 0) continue;
        if (rsize == 0) break;
        if (Utils::eq(row.first[column], value)) continue;
        float sum = 0;
        int pairs = 0;
        for (int c1 = 0; c1 < (int)classes.size()-1; ++c1) {
            for (int c2 = c1+1; c2 < (int)classes.size(); ++c2) {
                //find size of X1 and X2
                float nX1 = rightFreq[c1] + leftFreq[c1];
                float nX2 = rightFreq[c2] + leftFreq[c2];
                //since attribute values can only be "left" or "right" only two terms in summation
                float radicand = Utils::sqr(sqrt(1.*rightFreq[c2]/nX2) - sqrt(1.*rightFreq[c1]/nX1))+
                                 Utils::sqr(sqrt(1.*leftFreq[c2]/nX2) - sqrt(1.*leftFreq[c1]/nX1));

                sum += sqrt(radicand);
                pairs++;
            }
        }
        float gain = sum/pairs;
        if (gain > best_gain) {
            best_gain = gain;
            best_question = {column, row.first[column]};
        }
    }
}

void new_hellinger_split_discrete(const Rows &rows, int column,
                                  float &best_gain, Question &best_question) {
                                                  // +1 for missing values
    std::vector<std::vector<float>> attrClassCnts(attrValues[column].size()+1,
                                                  std::vector<float>(classes.size()));
    for (const auto &row: rows) {
        float value = row.first[column];
        if (value == MISSING) {
            value = attrValues[column].size();
        }
        attrClassCnts[(int)value][(int)row.first.back()] += row.second;
    }
    int legitChildren = 0;
    float total = 0.0f;
    for (int i = 0; i < (int)attrValues[column].size(); ++i) {
        float size = 0.0f;
        for (int j = 0; j < (int)classes.size(); ++j) {
            size += attrClassCnts[i][j];
            total += attrClassCnts[i][j];
        }
        if (size > 2 || Utils::eq(size, 2)) { // 2 should maybe be a parameter somewhere but I can't be bothered
            ++legitChildren;
        }
    }
    if (legitChildren < 2) {
        // no branching, don't process or update gain/question
        return;
    }
    float toSub = Utils::accumulate(attrClassCnts.back());

    float sum = 0;
	int pairs = 0;
    for (int c1 = 0; c1 < (int)classes.size()-1; ++c1) {
        for (int c2 = c1+1; c2 < (int)classes.size(); ++c2) {
            float nX1 = 0, nX2 = 0;
			for(int k = 0; k < (int)attrValues[column].size(); ++k){
				nX1+=attrClassCnts[k][c1];
				nX2+=attrClassCnts[k][c2];
			}
			float radicand = 0;
			for(int k = 0; k < (int)attrValues[column].size(); ++k){
				float nX1j = attrClassCnts[k][c1];
				float nX2j = attrClassCnts[k][c2];    

				radicand += Utils::sqr((sqrt(1.*nX1j/nX1) - sqrt(1.*nX2j/nX2)));
			}
			sum += sqrt(radicand);
			pairs++;
        }
    }
    float gain = (1-toSub/(toSub+total))*(sum/pairs);
    if (gain > best_gain) {
        best_gain = gain;
        best_question = {column, 0};
    }
}

void find_best_split_C45_hellinger(Rows &rows,
                                   float &best_gain, Question &best_question,
                                   int, bool) {
    best_gain = 0;
    int n_features = rows[0].first.size() - 1;
    for (int column = 0; column < n_features; ++column) {
        if (isContinuous[column]) {
            new_hellinger_split_continuous(rows, column, best_gain, best_question);
        } else {
            new_hellinger_split_discrete(rows, column, best_gain, best_question);
        }
    }
}
/* </hellinger> */

template<bool IGR>
void IG_split_continuous(const Rows &rows, int column, float beforeEntropy,
                         float &best_gain, Question &best_question, int max_buckets, bool c45) {
    std::set<float> values;
    if (max_buckets != 0) {
        float min = mins[column], max = maxs[column];
        float step = (max-min)/max_buckets;
        for (int bucket = 0; bucket <= max_buckets; ++bucket) {
            values.insert(bucket*step+min);
        }
    } else {
        values = feature_values(column, rows);
    }
    int size = 2+1; // +1 za missing
    std::vector<std::vector<float>> splits(size, std::vector<float>(classes.size()+1, 0)); // +1 za broj redova u tom splitu
    for (float value: values) {
        Question question = {column, value};
        for (auto &split: splits) for (float &e: split) e = 0;
        partition_class_histogram(rows, question, c45, splits);
        float gain;
        if constexpr (IGR) {
            gain = new_informationGainRatio(rows.size(), splits, beforeEntropy);
        } else {
            gain = new_informationGain(rows.size(), splits, beforeEntropy);
        }
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

template<bool IGR>
void IG_split_discrete(const Rows &rows, int column, float beforeEntropy,
                       float &best_gain, Question &best_question, bool c45) {
    int size = attrValues[column].size()+1; // +1 za missing
    std::vector<std::vector<float>> splits(size, std::vector<float>(classes.size()+1, 0)); // +1 za broj redova u tom splitu
    for (int value = 0; value < (int)attrValues[column].size(); ++value) {
        Question question = {column, 0.0f}; // value is unused
        for (auto &split: splits) for (float &e: split) e = 0;
        partition_class_histogram(rows, question, c45, splits);
        float gain;
        if constexpr (IGR) {
            gain = new_informationGainRatio(rows.size(), splits, beforeEntropy);
        } else {
            gain = new_informationGain(rows.size(), splits, beforeEntropy);
        }
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

template<bool IGR>
void find_best_split_C45(Rows &rows,
                            float &best_gain, Question &best_question,
                            int max_buckets, bool c45) {
    best_gain = 0;
    float beforeEntropy = entropy(rows);
    int n_features = rows[0].first.size() - 1;
    for (int column = 0; column < n_features; ++column) {
        if (isContinuous[column]) {
            IG_split_continuous<IGR>(rows, column, beforeEntropy, best_gain, best_question, max_buckets, c45);
        } else {
            IG_split_discrete<IGR>(rows, column, beforeEntropy, best_gain, best_question, c45);
        }
    }
}

template<void (*split_function)(Rows &, float &, Question &, int, bool)>
Node *build_CART_tree(Rows &&rows, int depth, int max_buckets) {
    float gain; 
    Question question;
    if (depth == 0) {
        return Leaf(rows);
    }
    split_function(rows, gain, question, max_buckets, false);
    if (Utils::eq(gain, 0)) {
        return Leaf(rows);
    }
    Rows true_rows, false_rows;
    float total_weight = 0.0f;
    partition(rows, question, false, true_rows, false_rows, total_weight);

    Node *true_branch = build_CART_tree<split_function>(std::move(true_rows), depth-1, max_buckets);
    Node *false_branch = build_CART_tree<split_function>(std::move(false_rows), depth-1, max_buckets);
    return DecisionNodeContinuous(false_branch, true_branch, question);
}

template<void (*split_function)(Rows &, float &, Question &, int, bool)>
Node *build_C45_tree(Rows &&rows, int depth, int max_buckets) {
    float gain; 
    Question question;
    if (depth == 0) {
        return Leaf(rows);
    }
    split_function(rows, gain, question, max_buckets, true);
    if (Utils::eq(gain, 0)) {
        return Leaf(rows);
    }
    std::vector<Rows> splits;
    Rows missing;
    float total_weight = 0.0f;
    if (isContinuous[question.column]) {
        splits.resize(2);
        missing = partition(rows, question, true, splits[1], splits[0], total_weight);
    } else {
        missing = partition(rows, question, true, splits, total_weight);
    }
    int total_rows = rows.size() - missing.size();
    for (const auto &row: missing) {
        for (auto &split: splits) {
            split.push_back(row);
            split.back().second *= (float)split.size() / total_rows;
        }
    }

    Node *node = new Node();
    node->question = question;
    node->total_weight = total_weight;
    for (auto &child_rows: splits) {
        if (child_rows.size() == 0) {
            node->children.push_back(Leaf(rows));
        } else {
            node->children.push_back(build_C45_tree<split_function>(std::move(child_rows), depth-1, max_buckets));
        }
    }
    return node;
}

void print_tree(Node *node, bool C45, const std::string &spacing="") {
    if (node->isLeaf) {
        printf("Predict {");
        bool comma = false;
        for (int i = 0; i < (int)classes.size(); ++i) { // should always be 2
            if (node->predictions[i] == 0) continue;
            if (C45) {
                printf("%s%s: %f", (comma?", ":""), classes[i].c_str(), node->predictions[i]);
            } else {
                printf("%s%s: %d", (comma?", ":""), classes[i].c_str(), (int)node->predictions[i]);
            }
            comma = true;
        }
        printf("}\n");
        return;
    }
    printf("\n%s", spacing.c_str());
    if (C45) {
        if (isContinuous[node->question.column]) {
            printf("if %s >= %.3f: ", attrNames[node->question.column].c_str(), node->question.value);
            print_tree(node->children[1], C45, spacing+"| ");
            printf("%selse: ", spacing.c_str());
            print_tree(node->children[0], C45, spacing+"| ");
        } else {
            printf("switch %s:\n", attrNames[node->question.column].c_str());
            for (int child = 0; child < (int)node->children.size(); ++child) {
                printf("%s| case %s: ", spacing.c_str(), attrValues[node->question.column][child].c_str()); // TODO: store more info about the node for better prints
                print_tree(node->children[child], C45, spacing+"| | ");
            }
        }
    } else {
        printf("if %s ", attrNames[node->question.column].c_str());
        if (isContinuous[node->question.column]) {
            printf(">= %.3f: ", node->question.value);
        } else {
            printf("== %s: ", attrValues[node->question.column][(int)node->question.value].c_str());
        }
        print_tree(node->children[1], C45, spacing+"| ");
        printf("%selse: ", spacing.c_str());
        print_tree(node->children[0], C45, spacing+"| ");
    }
}

std::vector<float> laplace(const std::vector<float> predictions) {
    const float total = Utils::accumulate(predictions);
    std::vector<float> probs;
    for (int e: predictions) {
        probs.push_back(1.*(e+1)/(total + classes.size()));
    }
    return probs;
}

std::vector<float> classify(const Row &row, Node *node, bool c45 = false) {
    if (node->isLeaf) {
        return laplace(node->predictions);
    }
    if (Utils::eq(row.first[node->question.column], MISSING)) {
        // if the value is missing return a.seconded average of all children
        std::vector<float> distribution(classes.size());
        for (const auto &child: node->children) {
            std::vector<float> returned = classify(row, child, c45);
            for (int i = 0; i < (int)returned.size(); ++i) {
                distribution[i] += child->total_weight / node->total_weight * returned[i];
            }
        }
        return distribution;
    }
    return classify(row, node->children[node->question.match(row, c45)], c45);
}

void print_leaf(const std::vector<int> &counts, int total) {
    printf("{");
    bool comma = false;
    for (int i = 0; i < (int)classes.size(); ++i) {
        float p = 1.*counts[i]/total*100;
        if (counts[i] == 0) continue;
        printf("%s%s: %.2f%%", (comma?", ":""), classes[i].c_str(), p);
        comma = true;
    }
    printf("}\n");
}

std::vector<std::string> parseLine(const std::string &line, char delimiter=',') {
    std::vector<std::string> splits;
    std::string cur;
    for (char c: line) {
        if (c == delimiter) {
            Utils::trim(cur);
            splits.push_back(cur);
            cur = "";
        } else {
            cur.push_back(c);
        }
    }
    Utils::trim(cur);
    if (!cur.empty()) {
        splits.push_back(cur);
    }
    return splits;
}

void makeDatastructure(const std::string &filePath) {
    std::ifstream fin(filePath);
    std::string line;
    bool getClasses = true;
    bool getAttrs = false;
    for (int lineno = 0; getline(fin, line); ++lineno) {
        Utils::trim(line);
        if (line.size() == 0) continue;
        if (line.back() == '.') line.pop_back();
        if (line == "|classes") {
            getClasses = true;
        } else if (getClasses) {
            classes = parseLine(line);
            if (classes.size() == 1) {
                fprintf(stderr, "%s\n%s\nWhat are you doing with just one class lol\n", line.c_str(), classes[0].c_str());
                exit(1);
            }
            getClasses = false;
            getAttrs = true;
        } else if (line == "|attributes") { // maybe unnecessary?
            getAttrs = true;
        } else if (getAttrs) {
            int colon = line.find(':');
            std::string attrName = line.substr(0, colon);
            attrNames.push_back(attrName);
            line = line.substr(colon+1);
            attrValues.push_back(parseLine(line));
            isContinuous.push_back(attrValues.back()[0] == "continuous");
        } else {
            fprintf(stderr, "Unsupported .names file, check line %d\n", lineno);
            fin.close();
            exit(1);
        }
    }
    fin.close();
}

void getData(const std::string &filestub, Rows &data) {
    makeDatastructure(filestub+".names");
    std::ifstream fin(filestub+".data");
    std::string line;
    mins.resize(attrValues.size(), NAN);
    maxs.resize(attrValues.size(), NAN);
    for (int lineno = 0; getline(fin, line); ++lineno) {
        auto values = parseLine(line);
        data.emplace_back(values.size(), 1.0f);
        for (int column = 0; column < (int)values.size()-1; ++column) {
            if (values[column] == "?") {
                data.back().first[column] = MISSING;
            } else if (isContinuous[column]) {
                data.back().first[column] = std::atof(values[column].c_str());
                if (std::isnan(mins[column])) {
                    mins[column] = data.back().first[column];
                    maxs[column] = data.back().first[column];
                }
                mins[column] = std::min(mins[column], data.back().first[column]);
                maxs[column] = std::max(maxs[column], data.back().first[column]);
            } else {
                auto x = std::find(attrValues[column].begin(),
                                   attrValues[column].end(),
                                   values[column]);
                if (x == attrValues[column].end()) {
                    fprintf(stderr, "Looking for \"%s\" in:\n", values[column].c_str());
                    for (const auto &v: attrValues[column]) {
                        fprintf(stderr, "%s, ", v.c_str());
                    }
                    fprintf(stderr, "\nCheck line %d\nThat don't exist bro.\n", lineno);
                    exit(1);
                }
                data.back().first[column] = x-attrValues[column].begin();
            }
        }
        auto x = std::find(classes.begin(), classes.end(), values.back());
        if (x == classes.end()) {
            fprintf(stderr, "That class don't exist bro.");
            exit(1);
        }
        data.back().first.back() = x-classes.begin();
    }
    fin.close();
}

float accuracy(int TP, int TN, int FP, int FN) {
    return 1.*(TP+TN)/(TP+TN+FP+FN);
}

float BA(int TP, int TN, int FP, int FN) {
    return (1.f*TP/(TP+FN) + 1.f*TN/(FP+TN))/2.f;
}

float fMeasure(int TP, int , int FP, int FN) {
    float precision = 1.*TP/(TP+FP);
    float sensitivity = 1.*TP/(TP+FN);
    return 2.*precision*sensitivity/(precision+sensitivity);
}

float MCC(int TP, int TN, int FP, int FN) {
    return (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
}

float AUC(int cl, const std::vector<std::vector<float>> &probs) {
    std::vector<float> pos, neg;
    for (const auto &prob: probs) {
        if ((int)prob.back() == cl) {
            pos.push_back(prob[cl]);
        } else {
            neg.push_back(prob[cl]);
        }
    }
    float critPair = 0;
    for (float p: pos) {
        for (float n: neg) {
            if (Utils::eq(p, n)) {
                critPair += 0.5;
            } else if (p > n) {
                critPair += 1.0;
            }
        }
    }
    return critPair / (pos.size() * neg.size());
}

std::vector<float> class_weights(const std::vector<std::vector<float>> &probs) {
    std::vector<float> weights(classes.size());
    for (const auto &prob: probs) {
        ++weights[(int)prob.back()];
    }
    for (int cl = 0; cl < (int)classes.size(); ++cl) {
       weights[cl] /= probs.size();
    }
    return weights;
}

std::vector<float> calcStats(const std::vector<int> &TP, const std::vector<int> &TN, 
               const std::vector<int> &FP, const std::vector<int> &FN,
               const std::vector<std::vector<float>> &probs) {
    std::vector<float (*)(int, int, int, int)> measures = {
        accuracy, BA, fMeasure, MCC
    };
    //fprintf(stderr, "\t\tACC\t\tBA\t\tF-1\t\tMCC\t\tAUC\n");
    const auto weights = class_weights(probs);
    std::vector<float> avgs(5);
    for (int cl = 0; cl < (int)classes.size(); ++cl) {
        //fprintf(stderr, "%8s:\t", classes[cl].c_str());
        for (int m = 0; m < (int)measures.size(); ++m) {
            float cur = measures[m](TP[cl], TN[cl], FP[cl], FN[cl]);
            //fprintf(stderr, "%8.6f\t", cur);
            avgs[m] += cur;
        }
        float cur = AUC(cl, probs);
        avgs[measures.size()] += cur*weights[cl];
        //fprintf(stderr, "%8.6f\n", cur);
    }
    fprintf(stderr, " Average:\t");
    for (int m = 0; m < (int)measures.size(); ++m) {
        avgs[m] /= classes.size();
        fprintf(stderr, "%8.6f\t", avgs[m]);
    }
    fprintf(stderr, "%8.6f\n", avgs[measures.size()]);
    return avgs;
}

Node *train(Rows &&data, int max_depth, int max_buckets, bool C45, bool hellinger, bool IGR) {
    Node *tree;
    if (hellinger) {
        if (C45) {
            fprintf(stderr, "Building C4.5 (HD) tree...\n");
            tree = build_C45_tree<find_best_split_C45_hellinger>(std::move(data), max_depth, max_buckets);
        } else {
            fprintf(stderr, "Building CART (HD) tree...\n");
            tree = build_CART_tree<find_best_split_hellinger>(std::move(data), max_depth, max_buckets);
        }
    } else {
        if (C45) {
            if (IGR) {
                fprintf(stderr, "Building C4.5 (IGR) decision tree...\n");
                tree = build_C45_tree<find_best_split_C45<true>>(std::move(data), max_depth, max_buckets);
            } else {
                fprintf(stderr, "Building C4.5 (IG) decision tree...\n");
                tree = build_C45_tree<find_best_split_C45<false>>(std::move(data), max_depth, max_buckets);
            }
        } else {
            fprintf(stderr, "Building CART (Gini) decision tree...\n");
            tree = build_CART_tree<find_best_split_google>(std::move(data), max_depth, max_buckets);
        }
    }
    return tree;
}

std::vector<float> test(const Rows &data, Node *tree, bool C45) {
    //print_tree(tree, C45);
    std::vector<int> TP(classes.size()), TN(classes.size()), FP(classes.size()), FN(classes.size());
    std::vector<std::vector<float>> probs;
    for (const auto &row: data) {
        int actual = row.first.back();
        probs.emplace_back(classify(row, tree, C45));
        //printf("Actual: %s. Predicted: ", classes[row.back().i].c_str()); print_leaf(hist, total);
        int prediction = std::max_element(probs.back().begin(), probs.back().end())-probs.back().begin();
        probs.back().push_back((int)row.first.back());
        for (int cl = 0; cl < (int)classes.size(); ++cl) {
            if (prediction == actual) {
                if (prediction == cl) ++TP[cl];
                else ++TN[cl];
            } else {
                if (prediction == cl) ++FP[cl];
                else ++FN[cl];
            }
        }
    }
    return calcStats(TP, TN, FP, FN, probs);
}

void cv_thread_runner(const Rows &data, int size, int passed, int max_depth, int max_buckets, bool C45, bool hellinger, bool IGR, std::vector<float> &avgs) {
    Rows training_data(data.size()-size), testing_data(size);
    for (int i = 0, tr = 0, te = 0; i < (int)data.size(); ++i) {
        if (i < passed) {
            training_data[tr++] = data[i];
        } else if (i < passed+size) {
            testing_data[te++] = data[i];
        } else {
            training_data[tr++] = data[i];
        }
    }
    Node *tree = train(std::move(training_data), max_depth, max_buckets, C45, hellinger, IGR);

    avgs = test(testing_data, tree, C45);
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "-h") == 0) {
        fprintf(stderr, "-h\t\tTo display this help message\n");
        fprintf(stderr, "-s <num>\tTo set the seed, 69 by default\n");
        fprintf(stderr, "-C45\t\tUse the C4.5 algorithm for building the decision tree. Will use CART if not specified\n");
        fprintf(stderr, "-IGR\t\tIf not using HD, use Information Gain Ratio instead of Information Gain.\n");
        fprintf(stderr, "-HD\t\tUse Hellinger distance\n");
        fprintf(stderr, "-f <filestem>\tSet filestem, expects <filestem>.data and <filestem>.names\n");
        // fprintf(stderr, "-t\tUse separate test file, expected <filestem>.test\n"); later
        fprintf(stderr, "-T <num>\tSet train to test ratio if no separate test file, 0.7 by default\n");
        fprintf(stderr, "-c <num1> <num2>\tCrosstrain <num1> times on <num2> folds\n");
        fprintf(stderr, "-b <num>\tSet max buckets for continuous values, 0 means don't use buckets. 0 by default\n");
        fprintf(stderr, "-d <num>\tSet max depth, 999 by default\n");
        fprintf(stderr, "-S\t\tShuffle dataset, off by default\n");
        fprintf(stderr, "-mt\t\tWhen doing crossvalidation, use one thread per fold\n");
        return 0;
    }
    int seed = 69;
    bool hellinger = false;
    bool IGR = false;
    bool C45 = false;
    bool shuffle = false;
    bool cv = false;
    bool multithread = false;
    std::string filestem; // ew std::string
    float train_to_test_ratio = 0.7f;
    int max_buckets = 0;
    int max_depth = 999;
    int numTimes = -1, numFolds = -1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-s") == 0) {
            if (++i < argc) {
                seed = std::atoi(argv[i]);
            } else {
                fprintf(stderr, "-s requires a value\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-HD") == 0) {
            hellinger = true;
        } else if (strcmp(argv[i], "-S") == 0) {
            shuffle = true;
        } else if (strcmp(argv[i], "-f") == 0) {
            if (++i < argc) {
                filestem = argv[i];
            } else {
                fprintf(stderr, "-f requires a value\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-T") == 0) {
            if (++i < argc) {
                train_to_test_ratio = std::atof(argv[i]);
            } else {
                fprintf(stderr, "-T requires a value\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (++i < argc) {
                max_buckets = std::atoi(argv[i]);
            } else {
                fprintf(stderr, "-b requires a value\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0) {
            if (++i < argc) {
                max_depth = std::atoi(argv[i]);
            } else {
                fprintf(stderr, "-d requires a value\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-C45") == 0) {
            C45 = true;
        } else if (strcmp(argv[i], "-IGR") == 0) {
            IGR = true;
        } else if (strcmp(argv[i], "-mt") == 0) {
            multithread = true;
        } else if (strcmp(argv[i], "-c") == 0) {
            cv = true;
            if (++i < argc) {
                numTimes = std::atoi(argv[i]);
            } else {
                fprintf(stderr, "-c two values\n");
                return 1;
            }
            if (++i < argc) {
                numFolds = std::atoi(argv[i]);
            } else {
                fprintf(stderr, "-c two values\n");
                return 1;
            }
        } else {
            fprintf(stderr, "Unrecognized commandline argument: %s\n", argv[i]);
            return 1;
        }
    }

    BEGIN_SESSION("HDTV"); 
    std::srand(seed);
    Rows data;
    getData(filestem, data);

    if (cv) {
        std::vector<float> avg(5);
        std::vector<std::vector<float>> avgs;
        avgs.resize(numFolds);
        for (int run = 0; run < numTimes; ++run) {
            std::random_shuffle(data.begin(), data.end());
            int passed = 0;
            std::vector<std::thread> ts(numFolds);
            for (int fold = 0; fold < numFolds; ++fold) {
                int size = data.size()/numFolds;
                if (fold == numFolds-1) size += data.size()%numFolds;
                //printf("Run: %d; Fold: %d %d %d\n", run, fold, data.size(), size);
                if (multithread) {
                    ts[fold] = std::thread(cv_thread_runner, std::cref(data), size, passed, max_depth, max_buckets, C45, hellinger, IGR, std::ref(avgs[fold]));
                } else {
                    cv_thread_runner(data, size, passed, max_depth, max_buckets, C45, hellinger, IGR, avgs[fold]);
                }
                passed += size;
            }
            for (int fold = 0; fold < numFolds; ++fold) {
                if (multithread) ts[fold].join();
                for (int k = 0; k < (int)avgs[fold].size(); ++k) {
                    avg[k] += avgs[fold][k];
                }
            }
        }
        fprintf(stderr, "\nTot.AVG:\t");
        for (float e: avg) {
            fprintf(stderr, "%8.6f\t", e/(numFolds*numTimes));
        }
    } else {
        if (shuffle) std::random_shuffle(data.begin(), data.end());
        int testCount = data.size()*(1-train_to_test_ratio);
        std::vector<float> avgs;
        cv_thread_runner(data, testCount, 0, max_depth, max_buckets, C45, hellinger, IGR, avgs);
    }
    printf("\n");
    END_SESSION();
    return 0;
}