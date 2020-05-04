#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <set>

#include <cstdio>
#include <cmath>
#include <cstring>

/* 
*** STO NEVALJA *** TODO

1. "weights" su nam krivo pretpostavljeni da trebaju biti 1
    - smanje se za rows koji kad splittamo na atributu s missing value
    
2. hellingera krivo izracunavamo jer smo retardirani
    - za diskretne slucajeve je kompletno krivo (kinda emulira continuous)
    - za "continumous" radi samo za 2 klase
    - NOTE: vAttrClassCnts[][] pribrojava sve weights u rows za trenutacni atribut gdje
            prvi [] odgovara vrijednosti atributa, a drugi klasi na kraju rowa

3. krucijalno, sve je presporo, a C4.5 je mozda najsporija stvar ikad napisana

4. dosta toga je c/p, moglo bi se nekako generalizirati da CART i C4.5 budu bazne podjele,
    a racunanje split gain-a da bude neovisno o tome

5. seemingly Information Gain Ratio konzistentno polucuje losijim rezultatima od Information Gain
    - ?? nema smisli

*/

// TODO: implementiraj hellinger C4.5
// TODO: implementiraj multi-class hellinger distance // lol
// TODO: sve osim stats u log file or sth
// TODO: dodaj cross validation
// TODO: handle missing data
// TODO: dodaj grid search za hiperparametre
// TODO: dodaj mogucnost spremanja stvorenog stabla
// TODO: fixati apsolutno sve da nije ovako fugly
// TODO: nemoj koristiti std::vector nego napravi nesto svoje sto ce imati countove potrebne za hellingera

// lol
// TODO: multi-core
// TODO: gpu?????

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

union Z { // to cast or not to cast, ma samo unija fuck it
    int i;
    float f;
    Z() {}
    Z(int x): i(x) {}
    Z(float x): f(x) {}
};

using Row = std::vector<Z>*; // first = actual podaci, second = weight tog reda
using Rows = std::vector<Row>;

int minorityClass; // TODO: get rid of these globals, legitimately just send them to everything that uses them
std::vector<int> totalHist;
std::vector<std::string> classes;
std::vector<std::string> attrNames;
std::vector<std::vector<std::string>> attrValues; // att[0] == {"b", "o", "x"}
std::vector<bool> isContinuous;
std::vector<float> mins, maxs;
std::vector<int> uniqueValueCount;

namespace Utils {
    const double epsilon = 1e-6;
    bool eq(double x, double y) {
        return fabs(x-y) <= epsilon;
    }

    Rows filter(const Rows &rows, int label) {
        // filter only for rows with label == `label`
        Rows new_rows;
        for (const auto &row: rows) {
            if (row->back().i == label) {
                new_rows.push_back(row);
            }
        }
        return new_rows;
    }

    int count(const Rows &rows, int label) {
        // count only for rows with label == `label`
        int ret = 0;
        for (const auto &row: rows) {
            if (row->back().i == label) {
                ++ret;
            }
        }
        return ret;
    }

    int count(const std::vector<std::vector<Z>> &rows, int label) {
        // count only for rows with label == `label`
        int ret = 0;
        for (const auto &row: rows) {
            if (row.back().i == label) {
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

    float log(float x) {
        if (eq(x, 0)) {
            return 0;
        } else {
            return std::log2f(x);
        }
    }
}

struct Question {
    int column;
    Z value; // TODO type pending // *mozda* nas nije briga

    int match(const Row &example, bool c45 = false) const { // a0: 5, a1: "burek", a2: 6
        // Compare the feature value in an example to the
        // feature value in this question.
        if (isContinuous[column]) {
            return (*example)[column].f >= value.f;
        }
        if (c45) { // returns the index of the value in attrValues[column]
            return (*example)[column].i;
        }
        return (*example)[column].i == value.i;
    }
};

std::vector<int> class_histogram(const Rows &rows) {
    /*
    count number of rows per class
    */
    std::vector<int> histogram(classes.size(), 0);
    for (const auto &row: rows) {
        int label = row->back().i;
        ++histogram[label];
    }
    return histogram;
}

struct Node {
    bool isLeaf = false;
    std::vector<Node *> children;
    Question question;
    std::vector<int> predictions;
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

float informationGain(const Rows &rows, const std::vector<Rows> &splits, float beforeEntropy) {
    float afterEntropy = 0;
    for (int i = 0; i < (int)splits.size(); ++i) {
        float weight = 1. * splits[i].size() / rows.size();
        afterEntropy += weight * entropy(splits[i]);
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

std::set<float> feature_values(int col, const Rows &rows) {
    // TODO: ne moze ostati float, ovisit ce o tipu attr
    // *mozda* nas to nije briga?
    std::set<float> values;
    for (const auto &row: rows) {
        if (isContinuous[col])
            values.insert((*row)[col].f);
        else
            values.insert((*row)[col].i);
    }
    return values;
} 

std::set<float> feature_values(int col, const std::vector<std::vector<Z>> &rows) {
    // TODO: ne moze ostati float, ovisit ce o tipu attr
    // *mozda* nas to nije briga?
    std::set<float> values;
    for (const auto &row: rows) {
        if (isContinuous[col])
            values.insert(row[col].f);
        else
            values.insert(row[col].i);
    }
    return values;
}

// u rows koliko ima labela == minorityClass takvih da je question.match(row)

// partition continuous
void partition(const Rows &rows, const Question &question, Rows &true_rows, Rows &false_rows) {
    for (const auto &row: rows) {
        if (question.match(row)) {
            true_rows.push_back(row);
        } else {
            false_rows.push_back(row);
        }
    }
}

// partition discrete
void partition(const Rows &rows, const Question &question, std::vector<Rows> &splits) {
    splits.resize(attrValues[question.column].size());
    for (const auto &row: rows) {
        splits[question.match(row, true)].push_back(row);
    }
}

/* <google indian> */
float info_gain(const Rows &left, const Rows &right,
                float current_uncertainty) {
    float p = 1. * left.size() / (left.size() + right.size());
    return current_uncertainty - p*gini(left) - (1-p)*gini(right);
}

void split_continuous(const Rows &rows, int column, float current_uncertainty,
                                float &best_gain, Question &best_question,
                                int max_buckets) {
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
    for (float value: values) {
        Question question = {column, value};
        Rows true_rows, false_rows;
        partition(rows, question, true_rows, false_rows);
        if (true_rows.size() == 0 || false_rows.size() == 0) {
            continue;
        }
        float gain = info_gain(true_rows, false_rows, current_uncertainty);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void split_discrete(const Rows &rows, int column, float current_uncertainty,
                               float &best_gain, Question &best_question) {
    for (int value = 0; value < (int)attrValues[column].size(); ++value) {
        Question question = {column, value};
        Rows true_rows, false_rows;
        partition(rows, question, true_rows, false_rows);
        if (true_rows.size() == 0 || false_rows.size() == 0) {
            continue;
        }
        float gain = info_gain(true_rows, false_rows, current_uncertainty);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void find_best_split_indian(const Rows &rows, float &best_gain, 
                            Question &best_question, int max_buckets) {
    best_gain = 0;
    float current_uncertainty = gini(rows);
    int n_features = rows[0]->size() - 1;
    for (int column = 0; column < n_features; ++column) {
        if (isContinuous[column]) {
            split_continuous(rows, column, current_uncertainty, best_gain, best_question, max_buckets);
        } else {
            split_discrete(rows, column, current_uncertainty, best_gain, best_question);
        }
    }
}
/* </google indian> */

/* <hellinger> */
float hellinger_distance(int lsize, int rsize, float tp, float tfvp, float tfwp) {
    float tfvn = lsize - tfvp;
    float tfwn = rsize - tfwp;
    float tn = (lsize + rsize) - tp;
    return Utils::sqr(std::sqrt(tfvp/tp) - std::sqrt(tfvn/tn))
         + Utils::sqr(std::sqrt(tfwp/tp) - std::sqrt(tfwn/tn));
}

void hellinger_split_continuous(const Rows &rows, int column, int tp,
                                float &best_gain, Question &best_question,
                                int max_buckets) {
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
    for (float value: values) {
        Question question = {column, value};
        int lsize = 0, rsize = 0;
        int tfvp = 0, tfwp = 0;
        for (const auto &row: rows) {
            if (question.match(row)) {
                ++lsize;
                if (row->back().i == minorityClass) ++tfvp;
            } else {
                ++rsize;
                if (row->back().i == minorityClass) ++tfwp;
            }
        }
        if (lsize == 0 || rsize == 0) {
            continue;
        }
        float gain = hellinger_distance(lsize, rsize, tp, tfvp, tfwp);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void hellinger_split_discrete(const Rows &rows, int column, int tp,
                               float &best_gain, Question &best_question) {
    for (int value = 0; value < (int)attrValues[column].size(); ++value) {
        Question question = {column, value};
        int lsize = 0, rsize = 0;
        int tfvp = 0, tfwp = 0;
        for (const auto &row: rows) {
            if (question.match(row)) {
                ++lsize;
                if (row->back().i == minorityClass) ++tfvp;
            } else {
                ++rsize;
                if (row->back().i == minorityClass) ++tfwp;
            }
        }
        if (lsize == 0 || rsize == 0) {
            continue;
        }
        float gain = hellinger_distance(lsize, rsize, tp, tfvp, tfwp);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void find_best_split_hellinger(const Rows &rows,
                               float &best_gain, Question &best_question,
                               int max_buckets) {
    best_gain = 0;
    float tp = Utils::count(rows, minorityClass);
    int n_features = rows[0]->size() - 1;
    for (int column = 0; column < n_features; ++column) {
        if (isContinuous[column]) {
            hellinger_split_continuous(rows, column, tp, best_gain, best_question, max_buckets);
        } else {
            hellinger_split_discrete(rows, column, tp, best_gain, best_question);
        }
    }
}
/* </hellinger> */

void IG_split_continuous(const Rows &rows, int column, float beforeEntropy,
                         float &best_gain, Question &best_question, int max_buckets) {
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
    for (float value: values) {
        Question question = {column, value};
        Rows true_rows, false_rows;
        partition(rows, question, true_rows, false_rows);
        float gain = informationGain(rows, {false_rows, true_rows}, beforeEntropy);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void IG_split_discrete(const Rows &rows, int column, float beforeEntropy,
                       float &best_gain, Question &best_question) {
    for (int value = 0; value < (int)attrValues[column].size(); ++value) {
        Question question = {column, value}; // value is unused
        std::vector<Rows> splits;
        partition(rows, question, splits);
        float gain = informationGain(rows, splits, beforeEntropy);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void find_best_split_C45_IG(const Rows &rows,
                            float &best_gain, Question &best_question,
                            int max_buckets) {
    best_gain = 0;
    float beforeEntropy = entropy(rows);
    int n_features = rows[0]->size() - 1;
    for (int column = 0; column < n_features; ++column) {
        if (isContinuous[column]) {
            IG_split_continuous(rows, column, beforeEntropy, best_gain, best_question, max_buckets);
        } else {
            IG_split_discrete(rows, column, beforeEntropy, best_gain, best_question);
        }
    }
}

void find_best_split_C45_IGR(); // TODO

void find_best_split_hellinger_IGR(); // TODO

template<void (*split_function)(const Rows &, float &, Question &, int)>
Node *build_CART_tree(const Rows &rows, int depth, int max_buckets) {
    float gain; 
    Question question;
    if (depth == 0) {
        return Leaf(rows);
    }
    // nez
    split_function(rows, gain, question, max_buckets);
    if (Utils::eq(gain, 0)) {
        return Leaf(rows);
    }
    Rows true_rows, false_rows;
    partition(rows, question, true_rows, false_rows);

    Node *true_branch = build_CART_tree<split_function>(true_rows, depth-1, max_buckets);
    Node *false_branch = build_CART_tree<split_function>(false_rows, depth-1, max_buckets);
    return DecisionNodeContinuous(false_branch, true_branch, question);
}

// TODO: finish this
//     - implement the split functions better
//     - make everything expect an arbitrary amount of children
//     - make the questions discern between left/right and multichildren
template<void (*split_function)(const Rows &, float &, Question &, int)>
Node *build_C45_tree(const Rows &rows, int depth, int max_buckets) {
    float gain; 
    Question question;
    if (depth == 0) {
        return Leaf(rows);
    }
    split_function(rows, gain, question, max_buckets);
    
    if (Utils::eq(gain, 0)) {
        return Leaf(rows);
    }
    std::vector<Rows> splits;
    if (isContinuous[question.column]) {
        splits.resize(2);
        partition(rows, question, splits[1], splits[0]);
    } else {
        partition(rows, question, splits);
    }

    Node *node = new Node();
    node->question = question;
    for (const auto &child_rows: splits) {
        if (child_rows.size() == 0) {
            node->children.push_back(Leaf(rows));
        } else {
            node->children.push_back(build_C45_tree<split_function>(child_rows, depth-1, max_buckets));
        }
    }
    return node;
}

/* TODO: finish later
struct NekaStrukturica {
    //lol[5][3]; // za atribut 5 cija je vrijednost >= BUCKET[0.3] koliko ima minority klasa?
    std::vector<int> count;
    Z min, max;
};

auto nez(Rows &rows) {
    std::vector<NekaStrukturica> lol(rows[0].size()-1);
    for (int column = 0; column < (int)rows[0].size()-1; ++column) {
        sort(rows.begin(), rows.end(), [column](const Row &a, const Row &b){
            if (isContinuous[column]) {
                return a[column].f < b[column].f;
            }
            return a[column].i < b[column].i;
        });
        if (isContinuous[column]) {
            lol[column].min = rows[0][column];
            lol[column].max = rows.back()[column];
            lol[column].count.resize(BUCKETS);
            fprintf(stderr, "\ncol: %d; min: %.2f; max: %.2f;\n", column, lol[column].min.f, lol[column].max.f);
        } else {
            lol[column].min = rows[0][column];
            lol[column].max = rows.back()[column];
            lol[column].count.resize(lol[column].max.i - lol[column].min.i + 1);
            fprintf(stderr, "col: %d; min: %d; max: %d; size: %d\n", column, lol[column].min.i, lol[column].max.i, lol[column].count.size());
        }
        float prevf = 0;
        if (isContinuous[column]) {
            for (int bucket = 0; bucket < BUCKETS; ++bucket) {
                float f = rows[row][column].f-prevf;
                float step = (lol[column].max.f-lol[column].min.f)/BUCKETS;
                if (row == 0 || f > step) {
                    prevf = f;
                    lol[column].count[bucket++] = rows.size()-row;
                    fprintf(stderr, "row: %d; bucket: %d (%.2f); count: %d;\n", row, bucket-1, prevf, lol[column].count[bucket-1]);
                }
            }
        } else {
            for (int row = 0, bucket = 0; row < (int)rows.size(); ++row) {
                if (row == 0 || (rows[row][column].i != rows[column][row-1].i)) {
                    lol[column].count[rows[row][column].i] = rows.size()-row;
                    fprintf(stderr, "row: %d; index: %d; count: %d;\n", row, rows[row][column].i, lol[column].count[rows[row][column].i]);
                }
            }
        }
    }
    return lol;
}
*/

void print_tree(Node *node, const std::string &spacing="") {
    if (node->isLeaf) {
        printf("%sPredict {", spacing.c_str());
        bool comma = false;
        for (int i = 0; i < (int)classes.size(); ++i) { // should always be 2
            if (node->predictions[i] == 0) continue;
            printf("%s%s: %d", (comma?", ":""), classes[i].c_str(), node->predictions[i]);
            comma = true;
        }
        printf("}\n");
        return;
    }
    printf("%s", spacing.c_str());
    printf("Is %s ", attrNames[node->question.column].c_str());
    if (isContinuous[node->question.column]) {
        printf(">= %.3f?\n", node->question.value.f);
    } else {
        printf("== %s?\n", attrValues[node->question.column][(int)node->question.value.i].c_str());
    }

    for (int child = 0; child < (int)node->children.size(); ++child) {
        printf("%s--> %d:\n", spacing.c_str(), child); // TODO: store more info about the node for better prints
        print_tree(node->children[child], spacing+"  ");
    }
}

std::vector<int> classify(const Row &row, Node *node, bool c45 = false) {
    if (node->isLeaf) {
        return node->predictions;
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

void getData(const std::string &filestub, std::vector<std::vector<Z>> &data) {
    makeDatastructure(filestub+".names");
    std::ifstream fin(filestub+".data");
    std::string line;
    mins.resize(attrValues.size(), NAN);
    maxs.resize(attrValues.size(), NAN);
    for (int lineno = 0; getline(fin, line); ++lineno) {
        auto values = parseLine(line);
        data.emplace_back(values.size());
        for (int i = 0; i < (int)values.size()-1; ++i) {
            if (values[i] == "?") {
                data.pop_back();
                break;
            }
            if (isContinuous[i]) {
                data.back()[i].f = std::atof(values[i].c_str());
                if (std::isnan(mins[i])) {
                    mins[i] = data.back()[i].f;
                    maxs[i] = data.back()[i].f;
                }
                mins[i] = std::min(mins[i], data.back()[i].f);
                maxs[i] = std::max(maxs[i], data.back()[i].f);
            } else {
                auto x = std::find(attrValues[i].begin(),
                                   attrValues[i].end(),
                                   values[i]);
                if (x == attrValues[i].end()) {
                    fprintf(stderr, "Looking for \"%s\" in:\n", values[i].c_str());
                    for (const auto &v: attrValues[i]) {
                        fprintf(stderr, "%s, ", v.c_str());
                    }
                    fprintf(stderr, "\nCheck line %d\nThat don't exist bro.\n", lineno);
                    exit(1);
                }
                data.back()[i].i = x-attrValues[i].begin();
            }
        }
        auto x = std::find(classes.begin(),
                           classes.end(),
                           values.back());
        if (x == classes.end()) {
            fprintf(stderr, "That class don't exist bro.");
            exit(1);
        }
        data.back().back().i = x-classes.begin();
    }
    fin.close();

    uniqueValueCount.resize(data[0].size()-1);
    for (int column = 0; column < (int)data[0].size()-1; ++column) {
        uniqueValueCount[column] = feature_values(column, data).size();
    }

    if (classes.size() == 2) { // TODO: bilo bi lipo da radi i za vise od 2 klasice
        totalHist.resize(2);
        totalHist[0] = Utils::count(data, 0);
        totalHist[1] = data.size()-totalHist[0];
    }
    minorityClass = min_element(totalHist.begin(), totalHist.end())-totalHist.begin();
}

void makeDataset(std::vector<std::vector<Z>> &data, unsigned int offset, Rows &dataset) {
    for (int i = offset; i < (int)(offset+dataset.size()); ++i) {
        dataset[i-offset] = &data[i];
    }
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

float AUC(int cl, const std::vector<std::vector<double>> &probs) {
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

std::vector<double> class_weights(const std::vector<std::vector<double>> &probs) {
    std::vector<double> weights(classes.size());
    for (const auto &prob: probs) {
        ++weights[(int)prob.back()];
    }
    for (int cl = 0; cl < (int)classes.size(); ++cl) {
        weights[cl] /= probs.size();
    }
    return weights;
}

float avgAUC(const std::vector<std::vector<double>> &probs) {
    float ret = 0;
    const auto weights = class_weights(probs);
    for (int cl = 0; cl < (int)classes.size(); ++cl) {
        double cur = AUC(cl, probs);
        fprintf(stderr, "%d: %f %f\n", cl, cur, weights[cl]);
        ret += cur * weights[cl];
    }
    return ret;
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "-h") == 0) {
        fprintf(stderr, "-h\tTo display this help message\n");
        fprintf(stderr, "-s\tTo set the seed, 69 by default\n");
        fprintf(stderr, "-C45\tuse the C4.5 algorithm for building the decision tree. Will use CART if not specified\n");
        fprintf(stderr, "-HD\tuse Hellinger distance\n");
        fprintf(stderr, "-f\tset filestem, expects <filestem>.data and <filestem>.names\n");
        // fprintf(stderr, "-t\tuse separate test file, expected <filestem>.test\n"); later
        fprintf(stderr, "-T\tset train to test ratio if no separate test file, 0.7 by default\n");
        fprintf(stderr, "-b\tset max buckets for continuous values, 0 means don't use buckets. 0 by default\n");
        fprintf(stderr, "-d\tset max depth, 999 by default\n");
        fprintf(stderr, "-S\tshuffle dataset, off by default\n");
        // fprintf(stderr, "-c\tset cross validation\n"); later
        return 0;
    }
    int seed = 69;
    bool hellinger = false;
    bool C45 = false;
    bool shuffle = false;
    std::string filestem; // ew std::string
    float train_to_test_ratio = 0.7f;
    int max_buckets = 0;
    int max_depth = 999;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-s") == 0) {
            if (++i < argc) {
                seed = std::atoi(argv[i]);
            } else {
                fprintf(stderr, "-s requires a value");
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
                fprintf(stderr, "-f requires a value");
                return 1;
            }
        } else if (strcmp(argv[i], "-T") == 0) {
            if (++i < argc) {
                train_to_test_ratio = std::atof(argv[i]);
            } else {
                fprintf(stderr, "-T requires a value");
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (++i < argc) {
                max_buckets = std::atoi(argv[i]);
            } else {
                fprintf(stderr, "-b requires a value");
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0) {
            if (++i < argc) {
                max_depth = std::atoi(argv[i]);
            } else {
                fprintf(stderr, "-d requires a value");
                return 1;
            }
        } else if (strcmp(argv[i], "-C45") == 0) {
            C45 = true;
        }
    }

    BEGIN_SESSION("HDTV"); 
    std::srand(seed);
    std::vector<std::vector<Z>> data;
    getData(filestem, data);

    Rows training_data, testing_data;
    if (shuffle) std::random_shuffle(data.begin(), data.end());
    int trainCount = data.size()*train_to_test_ratio;
    training_data.resize(trainCount);
    testing_data.resize(data.size()-trainCount);
    makeDataset(data, 0, training_data);
    makeDataset(data, trainCount, testing_data);
    
    //auto nn = nez(training_data);
    std::random_shuffle(training_data.begin(), training_data.end()); // unsort

    Node *tree;
    if (hellinger) { // TODO: add C45 hellinger
        fprintf(stderr, "Building Hellinger tree...\n");
        tree = build_CART_tree<find_best_split_hellinger>(training_data, max_depth, max_buckets);
        //tree = build_hellinger_tree(training_data, max_depth, max_buckets);
    } else {
        if (C45) {
            fprintf(stderr, "Building C4.5 decision tree...\n");
            tree = build_C45_tree<find_best_split_C45_IG>(training_data, max_depth, max_buckets);
        } else {
            fprintf(stderr, "Building decision tree...\n");
            tree = build_CART_tree<find_best_split_indian>(training_data, max_depth, max_buckets);
            //tree = build_decision_tree(training_data, max_depth, max_buckets);
        }
    }
    printf("\n");
    //print_tree(tree);
    float sum = 0;
    int TP = 0, TN = 0, FP = 0, FN = 0;
    int truers = 0;
    std::vector<std::vector<double>> probs;
    for (const auto &row: testing_data) {
        int actual = row->back().i;
        const auto hist = classify(row, tree, C45);
        float total = Utils::accumulate(hist);
        probs.emplace_back();
        for (int e: hist) {
            // LAPLACE
            probs.back().push_back(1.*(e+1)/(total + classes.size()));
            //printf("%f ", probs.back().back());
        }
        probs.back().push_back(row->back().i);
        //printf("%f\n", probs.back().back());
        //printf("Actual: %s. Predicted: ", classes[row.back().i].c_str()); print_leaf(hist, total);
        int prediction = std::max_element(hist.begin(), hist.end())-hist.begin();
        sum += hist[actual]/total;
        if (totalHist.size() == 2) {
            if (prediction == actual) {
                ++truers;
                if (prediction == minorityClass) ++TP;
                else ++TN;
            } else {
                if (prediction == minorityClass) ++FP;
                else ++FN;
            }
        }
    }
    printf("Average confidence:\t%8.5f%%\n", sum/testing_data.size()*100);
    printf("Accuracy:\t\t%8.5f%%\n", 1.*truers/testing_data.size()*100);
    if (totalHist.size() == 2) {
        printf("Actual accuracy:\t%8.5f%%\n", accuracy(TP, TN, FP, FN));
        printf("BA:\t\t\t%8.5f\n", BA(TP, TN, FP, FN));
        printf("f-measure:\t\t%8.5f\n", fMeasure(TP, TN, FP, FN));
        printf("avgAUC:\t\t\t%8.5f\n", avgAUC(probs));
    }
    END_SESSION();
    return 0;
}