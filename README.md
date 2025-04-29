# engr421-homework-09--spectral-clustering-solved
**TO GET THIS SOLUTION VISIT:** [ENGR421 Homework 09- Spectral Clustering Solved](https://www.ankitcodinghub.com/product/engr-421-dasc-521-introduction-to-machine-learning-homework-09-spectral-clustering-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;113902&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;ENGR421 Homework 09- Spectral Clustering Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
In this homework, you will implement a spectral clustering algorithm in Python. Here are the steps you need to follow:

1. You are given a two-dimensional data set in the file named hw09_data_set.csv, which contains 1000 data points generated randomly from nine bivariate Gaussian densities with the following parameters.

𝜇! = #++55..00( , Σ! = #+−00..86 −0.6( ,

+0.8 𝑁! = 100

𝜇” = #−+55..00( , Σ” = #++00..86 +0.6( ,

+0.8 𝑁” = 100

𝜇# = #−−55..00( , Σ# = #+−00..86 −0.6( ,

+0.8 𝑁# = 100

𝜇$ = #+−55..00( , Σ$ = #++00..86 +0.6( ,

+0.8 𝑁$ = 100

𝜇% = #++50..00( , Σ% = #++00..20 +0.0( ,

+1.2 𝑁% = 100

𝜇&amp; = #++05..00( , Σ&amp; = #++10..20 +0.0( ,

+0.2 𝑁&amp; = 100

𝜇’ = #−+50..00( , Σ’ = #++00..20 +0.0( ,

+1.2 𝑁’ = 100

𝜇( = #+−05..00( , Σ( = #++10..20 +0.0( ,

+0.2 𝑁( = 100

𝜇) = #++00..00( , Σ) = #++10..60 +0.0( ,

+1.6 𝑁) = 200

The given data points are shown in the following figure.

2. You should first calculate the Euclidean distances between the pairs of data points. The data point pairs with distance less than 𝛿 = 2.0 are considered as connected. Construct the matrix 𝐁 as follows:

𝑏*+ = 41, 5𝒙* − 𝒙+5″ &lt; 𝛿 0, otherwise.

𝑏** = 0

You should also visualize this connectivity matrix by drawing a line between two data points if they are connected. Your figure should be like the following figure. (20 points)

𝐋,-../01*2 = 𝐈 − 𝐃3!/”𝐁𝐃3!/”

print(L_symmetric[0:5, 0:5])

[[ 1. 0. -0.01277024 -0.01689343 -0.01277024]

[ 0. 1. -0.01683588 0. 0. ]

[-0.01277024 -0.01683588 1. 0. -0.01190476]

[-0.01689343 0. 0. 1. 0. ]

[-0.01277024 0. -0.01190476 0. 1. ]]

print(Z[0:5, 0:5])

[[ 0.02492986 -0.03008423 -0.00946604 0.05116243 0.0229564 ]

[ 0.01590745 -0.02367529 -0.00235798 0.037579 0.0141235 ]

[ 0.02601003 -0.03273215 -0.00886993 0.05526008 0.02411414]

[ 0.02147879 -0.02190107 -0.01054254 0.03633024 0.0175098 ] [ 0.02624211 -0.03262591 -0.00925994 0.05524827 0.02432336]]

5. Run k-means clustering algorithm on 𝐙 matrix to find 𝐾 = 9 clusters. When initializing your algorithm, use the following rows of 𝐙 matrix for initial centroids: 242, 528, 570, 590, 648, 667, 774, 891, and 955. (20 points)

6. Draw the clustering result obtained by your spectral clustering algorithm by coloring each cluster with a different color. Your figure should be like the following figure. (20 points)

What to submit: You need to submit your source code in a single file (.py file) named as STUDENTID.py, where STUDENTID should be replaced with your 7-digit student number.

How to submit: Submit the file you created to Blackboard. Please follow the exact style mentioned and do not send a file named as STUDENTID.py. Submissions that do not follow these guidelines will not be graded.

Cheating policy: Very similar submissions will not be graded.
