#include <bits/stdc++.h>

const float TRAIN_TO_TEST_RATIO = 0.70f;
const int SEED = 69;
const int MAX_DEPTH = 999;
const bool HELLINGER = false;

union Z {
    int i;
    float f;
    Z() {}
    Z(int x): i(x) {}
    Z(float x): f(x) {}
};

typedef std::vector<Z> Row;
typedef std::vector<Row> Rows;

namespace Utils {
    const double epsilon = 1e-6;
    bool eq(double x, double y) {
        return fabs(x-y) <= epsilon;
    }

    Rows filter(const Rows &rows, int label) {
        // filter only for rows with label == `label`
        Rows new_rows;
        for (const auto &row: rows) {
            if (row.back().i == label) {
                new_rows.push_back(row);
            }
        }
        return new_rows;
    }

    int count(const Rows &rows, int label) {
        // filter only for rows with label == `label`
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
};

struct Question {
    int column;
    float value; // TODO type pending

    bool match(const Row &example) const {
        // Compare the feature value in an example to the
        // feature value in this question.
        float value = example[this->column].f;
        return value >= this->value;
        /* TODO: ovaj if bi trebao biti check
                 je li atribut continuous
        if (this->column == 1) {
            return value >= this->value;
        }
        return value == this->value;
        */
    }

    void print() {
        printf("Is att%d ", this->column);
        printf(">= %.3f?\n", this->value);
        /* TODO: ovaj if bi trebao biti check
                 je li atribut continuous
        if (this->column == 1) {
            printf(">= %d?\n", this->value);
            return;
        }
        printf("== %s?\n", colors[this->value]);
        */
    }
};

std::vector<int> class_histogram(const Rows &rows) {
    /*
    count number of rows per class
    */
    std::vector<int> histogram(16);
    for (const auto &row: rows) {
        int label = row.back().i;
        while (histogram.size() <= label) {
            histogram.resize(histogram.size()*2);
        }
        ++histogram[label];
    }
    return histogram;
}

struct Node {
    bool isLeaf = false;
    Node *left = nullptr, *right = nullptr;
    Question question;
    std::vector<int> predictions;

    Node(Node *l, Node *r, const Question &q) :
        left(l), right(r), question(q) {}

    Node(const Rows &rows) : isLeaf(true), predictions(class_histogram(rows)) {}

    static Node *DecisionNode(Node *l, Node *r, const Question &q) {
        return new Node(l, r, q);
    }

    static Node *Leaf(const Rows &rows) {
        return new Node(rows);
    }
};

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

inline std::set<float> feature_values(int col, const Rows &rows) {
    // TODO: ne moze ostati float, ovisit ce o tipu attr
    std::set<float> values;
    for (const auto &row: rows) {
        values.insert(row[col].f);
    }
    return values;
}

void partition(const Rows &rows, const Question &question,
               Rows &true_rows, Rows &false_rows) {
    for (const auto &row: rows) {
        if (question.match(row)) {
            true_rows.push_back(row);
        } else {
            false_rows.push_back(row);
        }
    }
}

float info_gain(const Rows &left, const Rows &right,
                float current_uncertainty) {
    float p = 1. * left.size() / (left.size() + right.size());
    return current_uncertainty - p*gini(left) - (1-p)*gini(right);
}

float hellinger_distance(const Rows &left, const Rows &right, float tp) {
    float tfwp = Utils::count(right, 0);
    float tfvp = Utils::count(left, 0);
    float tfwn = right.size() - tfwp;
    float tfvn = left.size() - tfvp;
    float tn = (left.size() + right.size()) - tp;
    return Utils::sqr(std::sqrt(tfvp/tp) - std::sqrt(tfvn/tn))
         + Utils::sqr(std::sqrt(tfwp/tp) - std::sqrt(tfwn/tn));
}

void find_best_split(const Rows &rows, float &best_gain, 
                     Question &best_question) {
    best_gain = 0;
    float current_uncertainty = gini(rows);
    float tp = Utils::count(rows, 0);
    int n_features = rows[0].size() - 1;
    for (int column = 0; column < n_features; ++column) {
        auto values = feature_values(column, rows);
        for (float value: values) {
            Question question = {column, value};
            Rows true_rows, false_rows;
            partition(rows, question, true_rows, false_rows);
            if (true_rows.size() == 0 || false_rows.size() == 0) {
                continue;
            }
            float gain;
            if (HELLINGER) {
                gain = hellinger_distance(true_rows, false_rows, tp);
            } else {
                gain = info_gain(true_rows, false_rows, current_uncertainty);
            }
            if (gain >= best_gain) {
                best_gain = gain;
                best_question = question;
            }
        }
    }
    best_question.print();
    std::cerr << best_gain << std::endl;
}

Node *build_tree(const Rows &rows, int depth=MAX_DEPTH) {
    float gain;
    Question question;
    if (depth == 0) {
        return Node::Leaf(rows);
    }
    find_best_split(rows, gain, question);
    if (Utils::eq(gain, 0)) {
        return Node::Leaf(rows);
    }
    Rows true_rows, false_rows;
    partition(rows, question, true_rows, false_rows);

    Node *true_branch = build_tree(true_rows, depth-1);
    Node *false_branch = build_tree(false_rows, depth-1);
    return Node::DecisionNode(true_branch, false_branch, question);
}

void print_tree(Node *node, const std::string &spacing="") {
    if (node->isLeaf) {
        printf("%sPredict {", spacing.c_str());
        bool comma = false;
        for (int i = 0; i < 2; ++i) {
            if (node->predictions[i] == 0) continue;
            printf("%s%d: %d", (comma?", ":""), i, node->predictions[i]);
            comma = true;
        }
        printf("}\n");
        return;
    }
    printf("%s", spacing.c_str());
    node->question.print();

    printf("%s--> True:\n", spacing.c_str());
    print_tree(node->left, spacing+"  ");
    printf("%s--> False:\n", spacing.c_str());
    print_tree(node->right, spacing+"  ");
}

std::vector<int> classify(const Row &row, Node *node) {
    if (node->isLeaf) {
        return node->predictions;
    }
    if (node->question.match(row)) {
        return classify(row, node->left);
    }
    return classify(row, node->right);
}

float print_leaf(const std::vector<int> &counts, int actual) {
    float total = std::accumulate(counts.begin(), counts.end(), 0);
    printf("{");
    bool comma = false;
    float ret;
    for (int i = 0; i < 2; ++i) {
        float p = counts[i]/total*100;
        if (i == actual) ret = p;
        if (counts[i] == 0) continue;
        printf("%s%d: %.2f%%", (comma?", ":""), i, p);
        comma = true;
    }
    printf("}\n");
    return ret;
}

void getData(const std::string &file, 
             Rows &training_data, Rows &testing_data) {
    Rows data;
    std::ifstream fin(file);
    for (int i = 0, b = true; b; ++i) {
        char c; data.emplace_back(6);
        b = (bool)(fin >> data[i][0].f >> c >> data[i][1].f >> c 
                       >> data[i][2].f >> c >> data[i][3].f >> c
                       >> data[i][4].f >> c >> data[i][5].i);
    }
    fin.close();

    std::random_shuffle(data.begin(), data.end());
    int trainCount = data.size()*TRAIN_TO_TEST_RATIO;
    training_data.resize(trainCount);
    testing_data.resize(data.size()-trainCount);
    std::copy_n(data.begin(), trainCount, training_data.begin());
    std::copy_n(data.rbegin(), data.size()-trainCount, testing_data.begin());
}

int main(int argc, char **argv) {
    std::srand(SEED);
    Rows training_data, testing_data;
    std::string dataFile; std::cin >> dataFile;
    getData(dataFile, training_data, testing_data);

    Node *tree = build_tree(training_data);
    printf("\n");
    print_tree(tree);
    float sum = 0;
    for (const auto &row: testing_data) {
        printf("Actual: %d. Predicted: ", row.back().i);
        sum += print_leaf(classify(row, tree), row.back().i);
    }
    printf("Average certainty: %.2f%%", sum/testing_data.size());
}
