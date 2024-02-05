CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

## I. Scan 

### 1. A Naive Paralle Scan


![naive_parallel](resources/naive_parallel.jpeg)

 * d = 1 → addition operation from 1 to n
 * d = 2 → addition operation from 2 to n
 * d = 3 → addition operation from 4 to n
 * ...
 * d = $log_2n$ → addition operation from ${2^{d-1}}$ to n

#### Algorithm:
---
*for d = 1 to $log_2n$\
&nbsp;&nbsp;&nbsp;&nbsp; for k = 0 to n \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if (k>=${2^{d-1}}$) output[k] += input[k - ${2^{d-1}}$] \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; else output[k] = input[k]*

--- 

It should be noted that with this algorithm in-place change could cause race condition, as shown below. \
1 <img src="resources/double_buffer1.png" width="40%" height="40%"> 2<img src="resources/double_buffer2.png" width="40%" height="40%">
swapping the order of two operations leads to different scan results. Thus we need two arrays, input and output. 

#### Work complexity 

$O(nlog_2n)$ addition operations.

As it's on GPU, for smaller array each pass $n$ operations can be performed in parallel. When the array size is larger than the number of available threads, it becomes quite inefficient. 

### 2. Work-efficient Parallel Scan 
The algorithm consists of two phases: up-sweep and down-sweep:

- up-sweep

<img src="resources/up-sweep.jpeg">\
- down-sweep 

<img src="resources/down-sweep.jpeg"> 


In up-sweep, every other prefix-sum is computed in place. The down-sweep completes the scan by distributing the accumulated sum (prefix sum) from each node to the adjacent nodes in the tree, effectively adding the adjacent number to every other prefix-sum.

#### Algorithm
---
// **up-sweep** 

for d = 0 to $log_2n$\
&nbsp;&nbsp;&nbsp;&nbsp; for k = 0 to n with step $2^{d+1}$ \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; x[k + $2^{d+1}$ - 1] += x[$2^{d}$ - 1] 

// **down-sweep**

**x[n-1]= 0**\
for d = $log_2n$ to 0\
&nbsp;&nbsp;&nbsp;&nbsp; for k = 0 to n with step $2^{d+1}$ \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; t = x[$2^{d}$ - 1]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; x[$2^{d}$ - 1] = x[$2^{d+1}$ - 1]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; x[$2^{d}$ - 1] += t<br>

--- 

#### Pay attention to: 
1. This algorithm builds a balanced tree for two-phase sweep. The array is divisible by the step size if the array size is power of 2. In the case of non-power-of-two array size, we need extra memory to account for the remaining. Analogous to `ilog2ceil` to compute ceiling value of $log_2n$, I wrote `idivjceil` to compute the ceiling value of $n/step$ where $step = 2^{d+1}$

## II. Stream Compaction 

Stream compaction is used to eliminate all the zero-value items in the input array. 
- CPU brute-force: adding every non-zero value in the input array to a separate output array, which cannot be parallelized because of the dependency on indexing. 
- On GPU with parallel scan: 
1) in another array, label each value in input array by 0 and 1 for zero and non-zero values, respectively. stored in array `label`
2) compute scan of the label array stored in array `scan`
3) for i = 0 to n\
    if label[i]!=0\
    output[scan[i]] = input[i]

All three steps can be parallelized on GPU as there's no dependency between each operation. 

### Pay attention to:
By calculating exclusive-scan of the label array, `scan[i]` corresponds to the index in the output for `input[i]` if `input[i]` is non-zero. The return value of stream compaction is the size of the compacted array. To obtain the size from `scan`, we need to look at two cases
1) the last item in `label` is zero -> size = scan[n-1]
2) the last item in `label` is non-zero -> size = scan[n-1] + 1

To handle both senarios more clearly, we pad the input array with a trailing zero. Therefore, no matter which case the original input falls into, we can always obtain size of the compacted array from `scan[n-1]`. 

