# Tips Optimization

TipsOptimizer is an instruction optimization algorithm that enhances LLM performance on specific tasks by summarizing some tips from the training set based on positive and negative examples, 
and adding them to the original system prompt.

The following table shows the performance of different methods on various tasks:

<table>
    <tr>
        <th></th>
        <th colspan="3">English</th>
        <th colspan="3">Chinese</th>
    <tr>
    <tr>
        <th></th>
        <th>GSM8K</th>
        <th>BBH(word_sorting)</th>
        <th>BBH(object_counting)</th>
        <th>THUNews</th>
        <th>CMMLU(nutrition)</th>
        <th>CMMLU(college_medical_statistics)</th>
    <tr>
    <tr>
        <td style="text-align:center">raw</td>
        <td style="text-align:center">96.1%</td>
        <td style="text-align:center">57.0%</td>
        <td style="text-align:center">97.0%</td>
        <td style="text-align:center">85.0%</td>
        <td style="text-align:center">93.2%</td>
        <td style="text-align:center">73.6%</td>
    <tr>
    <tr>
        <td style="text-align:center">OPRO</td>
        <td style="text-align:center">96.3%</td>
        <td style="text-align:center">56.0%</td>
        <td style="text-align:center">97.0%</td>
        <td style="text-align:center">84.0%</td>
        <td style="text-align:center">90.4%</td>
        <td style="text-align:center">75.5%</td>
    <tr>
    <tr>
        <td style="text-align:center">PromptAgent</td>
        <td style="text-align:center">95.7%</td>
        <td style="text-align:center">60.0%</td>
        <td style="text-align:center">97.0%</td>
        <td style="text-align:center">80.8%</td>
        <td style="text-align:center">95.9%</td>
        <td style="text-align:center">71.7%</td>
    <tr>
    <tr>
        <th style="text-align:center">TipsOptimizer</th>
        <th style="text-align:center">96.1%</th>
        <th style="text-align:center">64.0%</th>
        <th style="text-align:center">98.0%</th>
        <th style="text-align:center">90.0%</th>
        <th style="text-align:center">95.9%</th>
        <th style="text-align:center">83.0%</th>
    <tr>
</table>