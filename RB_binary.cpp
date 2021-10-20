//
//  RB.cpp
//  
//
//
//
// How to use this code?
// Step 1: Change N to the number of arms.
// Step 2: Create a text file with N*maxZ entries. The first maxZ entries are the reward function (f(1), f(2), ..., f(maxZ)) of the first arm, the second maxZ entried are the reward function of the second arm, ...
// Step 3: Run this code by direction the standard input to the text file.

#include <iostream>
#include <vector>
using namespace std;

const int maxZ = 20;
const int N = 100;
const int M  = 25;
const int d = 20;
const double beta = 0.99;
const int T = 300;
const int num_class = 4;    //there are four classes
double reward[N][maxZ];

struct arm{
    int index;
    int Z;
    double reward[maxZ];
};

struct decision{
    double tot_reward;
    vector< vector<bool> > selected; //selected[t][n] denotes whether the arm n should be activated at time (d-1-(t%d))
};

bool sort_desending_reward (arm i,arm j) { return (i.reward[i.Z] > j.reward[j.Z]); }

decision best_binary_tree(int remaining_depth, vector<arm> arm_list);

int main(){
    vector<arm> arm_list;
    
    for(int i = 0; i < N; i++){
        arm x;
        x.index = i;
        x.Z = 0;
        for(int z = 0; z < maxZ; z++)
            cin >> x.reward[z];
        arm_list.push_back(x);
    }
    
    decision best_arms;
    best_arms.selected.clear();
    double discount = 1;
    double R = 0;
    for(int t = 0; t < T; t++){
        if(best_arms.selected.size() == 0){    //compute the actions in the next d steps
            cout << t << endl;
            best_arms = best_binary_tree(d, arm_list);
            cout << best_arms.selected.size() << " " << best_arms.selected[0].size() << endl;
            
        }
        vector<bool> choice_for_this_slot = best_arms.selected.back();
        best_arms.selected.pop_back();
        
        for(int i = 0; i < N; i++){
            if( choice_for_this_slot[arm_list[i].index] == true){    //this arm is activated
                R += discount*arm_list[i].reward[arm_list[i].Z];
                arm_list[i].Z = 0;
            }else{
                arm_list[i].Z++;
                if(arm_list[i].Z > maxZ-1)
                    arm_list[i].Z = maxZ-1;
            }
        }
        discount *= beta;
    }
    
    cout << R << endl;
    return 1;
}


decision best_binary_tree(int remaining_depth, vector<arm> arm_list){
    decision option1, option2;
    vector<arm> list1, list2;
    vector<bool> choice1, choice2;
    double R1, R2;
    
    
    sort(arm_list.begin(), arm_list.end(), sort_desending_reward);
    
    //calculate the reward of option1: top M
    option1.selected.clear();
    list1 = arm_list;
    R1 = 0;
    for(int i = 0;i < N; i++)
        choice1.push_back(false);
    
    for(int i = 0; i < N; i++){
        if(i < M){  //the arm is activated;
            R1 += list1[i].reward[list1[i].Z];
            list1[i].Z = 0;
            choice1[list1[i].index] = true;
        }
        else{
            list1[i].Z++;
            if(list1[i].Z > maxZ)
                list1[i].Z = maxZ;
        }
    }
    if(remaining_depth > 1){
        option1 = best_binary_tree(remaining_depth-1, list1);
    }
    option1.tot_reward = option1.tot_reward*beta + R1;
    option1.selected.push_back(choice1);

    //calculate the reward of option2: top M-1 and #M+1
    option2.selected.clear();
    list2 = arm_list;
    R2 = 0;
    for(int i = 0;i < N; i++)
        choice2.push_back(false);
    
    for(int i = 0; i < N; i++){
        if(i < M-1 || i==M){  //the arm is activated;
            R2 += list2[i].reward[list2[i].Z];
            list2[i].Z = 0;
            choice2[list2[i].index] = true;
        }
        else{
            list2[i].Z++;
            if(list2[i].Z > maxZ)
                list2[i].Z = maxZ;
        }
    }
    if(remaining_depth > 1){
        option2 = best_binary_tree(remaining_depth-1, list2);
    }
    option2.tot_reward = option2.tot_reward*beta + R2;
    option2.selected.push_back(choice2);

    //pick the better one
    if(option1.tot_reward > option2.tot_reward)
        return option1;
    else
        return option2;

}
