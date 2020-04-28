#include <bits/stdc++.h>

typedef std::vector<int> Row;
typedef std::vector<Row> Rows;

// TODO: load iz fajla
// really only used for printing
char *attributes[] = {"Color", "Diameter"};
char *colors[] = {"Green", "Yellow", "Red"};
char *classes[] = {"Apple", "Grape", "Lemon"};

namespace Utils {
    const double epsilon = 1e-6;
    bool eq(double x, double y) {
        return fabs(x-y) <= epsilon;
    }
};

struct Question {
    int column;
    int value; // TODO type pending

    bool match(const Row &example) const {
        // Compare the feature value in an example to the
        // feature value in this question.
        int value = example[this->column];
        if (this->column == 1) {
            return value >= this->value;
        }
        return value == this->value;
    }

    void print() {
        printf("Is %s ", attributes[this->column]);
        if (this->column == 1) {
            printf(">= %d?\n", this->value);
            return;
        }
        printf("== %s?\n", colors[this->value]);
    }
};

std::vector<int> class_histogram(const Rows &rows) {
    /*
    count number of rows per class
    */
    std::vector<int> histogram(16);
    for (const auto &row: rows) {
        int label = row.back();
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
        left(l), right(r), question(q) {
        std::cerr << q.column << ' ' << q.value << "\n";
    }

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

inline std::set<int> feature_values(int col, const Rows &rows) {
    std::set<int> values;
    for (const auto &row: rows) {
        values.insert(row[col]);
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

void find_best_split(const Rows &rows, float &best_gain, 
                     Question &best_question) {
    best_gain = 0;
    float current_uncertainty = gini(rows);
    int n_features = rows[0].size() - 1;
    for (int column = 0; column < n_features; ++column) {
        auto values = feature_values(column, rows);
        for (int value: values) {
            Question question = {column, value};
            Rows true_rows, false_rows;
            partition(rows, question, true_rows, false_rows);
            if (true_rows.size() == 0 || false_rows.size() == 0) {
                continue;
            }

            float gain = info_gain(true_rows, false_rows, current_uncertainty);
            if (gain >= best_gain) {
                best_gain = gain;
                best_question = question;
            }
        }
    }
}

Node *build_tree(const Rows &rows) {
    float gain;
    Question question;
    find_best_split(rows, gain, question);
    if (Utils::eq(gain, 0)) {
        return Node::Leaf(rows);
    }
    Rows true_rows, false_rows;
    partition(rows, question, true_rows, false_rows);

    Node *true_branch = build_tree(true_rows);
    Node *false_branch = build_tree(false_rows);
    return Node::DecisionNode(true_branch, false_branch, question);
}

void print_tree(Node *node, const std::string &spacing="") {
    if (node->isLeaf) {
        printf("%sPredict {", spacing.c_str());
        bool comma = false;
        for (int i = 0; i < 3; ++i) {
            if (node->predictions[i] == 0) continue;
            printf("%s%s: %d", (comma?", ":""), classes[i], node->predictions[i]);
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

void print_leaf(const std::vector<int> &counts) {
    float total = std::accumulate(counts.begin(), counts.end(), 0);
    printf("{");
    bool comma = false;
    for (int i = 0; i < 3; ++i) {
        if (counts[i] == 0) continue;
        printf("%s%s: %.2f%%", (comma?", ":""), classes[i], counts[i]/total*100);
        comma = true;
    }
    printf("}\n");
}

// color (index), diameter, label (index)
Rows training_data = {
    {0, 3, 0}, // green, 3, apple
    {1, 3, 0}, // yellow, 3, apple
    {2, 1, 1}, // red, 1, grape
    {2, 1, 1}, // red, 1, grape
    {1, 3, 2}  // yellow, 3, lemon
};

Rows testing_data = {
    {0, 3, 0},
    {1, 4, 0},
    {2, 2, 1},
    {2, 1, 1},
    {1, 3, 2},
};

int main() {
    Node *tree = build_tree(training_data);
    print_tree(tree);
    for (int i = 0; i < 5; ++i) {
        printf("Actual: %s. Predicted: ", classes[testing_data[i][2]]);
        print_leaf(classify(testing_data[i], tree));
    }
}