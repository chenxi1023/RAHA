import math
import sys

def hard_attention(abstract, reference):
    abstract = abstract[:1000] if len(abstract) > 1000 else abstract
    reference = reference[:1000] if len(reference) > 1000 else reference

    prompt = f'''
Determine whether a reference paper is important to a focal paper based on the abstract. 
Return Import Index is "1" if it is important and "0" if it is not.
Don't repeat my inputs, just output the values.

Example as follows:
Input：
Focal paper abstract:abstract1
Reference paper abstract:reference1
Output:
0

Input
Focal paper abstract:{abstract}
Reference paper abstract:{reference}
Output:
'''
    return prompt

def prompt_difference(abstract, reference):
    abstract = abstract[:1000] if len(abstract) > 1000 else abstract
    try:
        reference = reference[:1000] if len(reference) > 1000 else reference
    except Exception as e:  # 捕获异常，并将其赋值给 e
        # print(reference)
        # print("发生异常：", e)  # 打印异常信息
        # sys.exit(1)  # 退出程序，并返回状态码 1 表示异常终止
        pass
    prompt = f'''
You are now tasked with assessing the disruptive potential in the research area of academic papers. 
Your approach involves contrasting the abstract of a focus paper with the abstracts of its cited references. 
No need to give me abstract's analysis, just output Contrast and Difference.

Focus paper abstract:{abstract}
Reference paper abstract:{reference}
Contrast and Difference:'''
    return prompt

def regression_generation(abstract, reference, d_index=None):
    abstract = abstract[:500] if len(abstract) > 500 else abstract
    reference = reference[:1500] if len(reference) > 1500 else reference
    # d_index = math.exp(d_index) - 1.01
    prompt = f'''
- Determine whether the dindex predicted in the previous epoch is high or low: [DINDEX]{d_index}[DINDEX]
- Abstract of Focus Paper: {abstract}
- Comparison with Reference Paper : {reference}
'''
    return prompt
def prompt_paper(input,d_index=None):
    input = input[:2000] if len(input) > 2000 else input
    prompt = f'''
    You are tasked with assessing the disruptive potential of academic papers. Your primary tool for this analysis is the Disruption Index, a metric ranging from -1 to 1. This index quantifies the level of innovation or breakthrough a paper represents. A higher positive value on the index indicates a significant breakthrough, while negative values suggest a lower level of innovation.
    Your analysis should focus on understanding and explaining the reasons behind the focus paper's disruptive nature. For this, you should consider its Disruption Index as a key indicator. The goal is to elucidate how and why the paper represents a breakthrough or lacks innovation, as indicated by its Disruption Index.
    Please provide a detailed analysis based on the contrast and differences between the focus paper and its references. Use the Disruption Index of the focus paper to guide your assessment. Pay special attention to the unique contributions or shortcomings of the focus paper in comparison to the referenced works.

    Details for Analysis:
    - Disruption Index of Focus Paper: {d_index}
    - Abstract of Focus Paper: {input}

    Based on the above information, analyze the reason for the disruptive nature (or lack thereof) of the focus paper. '''
    return prompt


def prompt_generation(abstract, reference, d_index=None):
    abstract = abstract[:500] if len(abstract) > 500 else abstract
    reference = reference[:2200] if len(reference) > 2000 else reference
    prompt = f'''
You are tasked with assessing the disruptive potential of academic papers. Your primary tool for this analysis is the Disruption Index, a metric ranging from -1 to 1. This index quantifies the level of innovation or breakthrough a paper represents. A higher positive value on the index indicates a significant breakthrough, while negative values suggest a lower level of innovation.
Your analysis should focus on understanding and explaining the reasons behind the focus paper's disruptive nature. For this, you should consider its Disruption Index as a key indicator. The goal is to elucidate how and why the paper represents a breakthrough or lacks innovation, as indicated by its Disruption Index.
Please provide a detailed analysis based on the contrast and differences between the focus paper and its references. Use the Disruption Index of the focus paper to guide your assessment. Pay special attention to the unique contributions or shortcomings of the focus paper in comparison to the referenced works.

Details for Analysis:
- Disruption Index of Focus Paper: {d_index}
- Abstract of Focus Paper: {abstract}
- Comparison with Reference Paper : {reference}

Based on the above information, analyze the reason for the disruptive nature (or lack thereof) of the focus paper. '''
    return prompt

def patent_importance(abstract, reference):
    # Truncate the abstract and reference to manageable lengths
    abstract = abstract[:1000] if len(abstract) > 1000 else abstract
    reference = reference[:1000] if len(reference) > 1000 else reference

    prompt = f'''
Assess the importance of a reference patent based on its abstract in relation to a focal patent. 
Return an Importance Index as "1" if it is important and "0" if it is not.
Do not repeat the inputs, only provide the evaluation.

Example as follows:
Input:
Focal Patent Abstract:abstract
Reference Patent Abstract:reference
Output:
0

Input:
Focal Patent Abstract:{abstract}
Reference Patent Abstract:{reference}
Output:
'''
    return prompt

def patent_contrast_difference(abstract, reference):
    abstract = abstract[:1000] if len(abstract) > 1000 else abstract
    reference = reference[:1000] if len(reference) > 1000 else reference

    prompt = f'''
You are tasked with analyzing the innovation gap and potential impact between patents. 
Your job is to contrast the abstract of a focal patent with the abstracts of its related patents. 
Avoid providing an analysis of the abstracts themselves; focus instead on the contrast and potential differences.

Focal Patent Abstract:{abstract}
Related Patent Abstract:{reference}
Contrast and Difference:'''
    return prompt

def patent_innovation_analysis(abstract, reference, d_index=None):
    abstract = abstract[:500] if len(abstract) > 500 else abstract
    reference = reference[:2200] if len(reference) > 2000 else reference

    prompt = f'''
You are tasked with evaluating the innovation level and potential breakthrough of patents. 
Your primary tool for this analysis is the Innovation Index, a metric ranging from -1 to 1. 
This index helps quantify the level of novelty and potential market disruption a patent represents. 
A higher positive value on the index indicates a significant breakthrough, while negative values suggest incremental or less novel innovations.
Please provide a detailed assessment based on the comparison between the focal patent and its related patents. 
Consider the Innovation Index of the focal patent to guide your analysis, focusing on the unique contributions or advancements it offers.

Details for Assessment:
- Last predicted Innovation Index of Focal Patent: {d_index}
- Abstract of Focal Patent: {abstract}
- Comparison with Related Patent: {reference}

Based on the above information, predict the innovation index of the focal patent. '''
    return prompt

def patent_disruption_analysis(abstract, reference, d_index=None):
    abstract = abstract[:500] if len(abstract) > 500 else abstract
    reference = reference[:2200] if len(reference) > 2000 else reference

    prompt = f'''
You are tasked with understanding the disruptive potential of patents. 
The Disruption Index, ranging from -1 to 1, serves as your primary tool for this analysis. 
This metric indicates the level of market disruption and novelty the patent may introduce. 
A higher positive value on the index indicates a significant market shift, while negative values suggest a more incremental innovation.
Your analysis should explore the reasons behind the patent's disruptive potential based on its Disruption Index. 
Focus on identifying and explaining the unique contributions or advancements the patent makes compared to existing ones.

Details for Analysis:
- Disruption Index of Focal Patent: {d_index}
- Abstract of Focal Patent: {abstract}
- Comparison with Related Patent: {reference}

Based on the above information, analyze the reasons behind the disruptive nature (or lack thereof) of the focal patent. '''
    return prompt