# Additional Results for Experiment
The results of the following three additional tables are used to support <b/>the second half of Observation 4)</b> and <b/>the second half of the corresponding analysis</b>, and <b/>the second half of the final summary paragraph</b> in <b/>Section Experimental Validation</b>.

This is the performance and greenness comparison of the proposed Coding-Fuse method and the original single model OSM. Among them in the table, the bold values indicate that Coding-Fuse is better than OSM, and the non-bold italic values indicate that Coding-Fuse is worse than OSM. It can be seen that the <b/>Coding-Fuse can achieve better performance than OSM at the same efficiency and hardware resource level on the code-related classification tasks in both the code embedding scenario and the finetune model scenario.</b>

![Additional Table I](https://github.com/SEOpenLab/ASE-2025/blob/main/Additional%20Result/A-CCD.jpg)
![Additional Table II](https://github.com/SEOpenLab/ASE-2025/blob/main/Additional%20Result/A-TDD.jpg)
![Additional Table III](https://github.com/SEOpenLab/ASE-2025/blob/main/Additional%20Result/A-CSD.jpg)

# Additional Results for Discussion
The <b/>Section Why effective?</b> in <b/>Section Discussion</b> only shows the comparison of the average entropy of the output feature vectors between Coding-Fuse and FMF in the CSD task. Here we supplement the entropy comparison on the remaining two tasks. 

<b/>As shown in Figures 1 and 2, on the remaining two tasks, the average entropy of the feature vectors output by Coding-Fuse is larger than that of FMF in most cases, which means that the informativeness of the feature vector output by Coding-Fuse is higher, which in turn leads to the performance advantage of the classifier trained by the Coding-Fuse method over FMF. This result also supports the conclusion in the original manuscript.</b>

![Additional Figure 1](https://github.com/SEOpenLab/ASE-2025/blob/main/Additional%20Result/D-CCD.png)
![Additional Figure 2](https://github.com/SEOpenLab/ASE-2025/blob/main/Additional%20Result/D-TDD.png)
