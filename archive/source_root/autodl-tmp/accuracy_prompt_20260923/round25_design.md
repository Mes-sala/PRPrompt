# 第二十五轮：蒸馏首选类别与其余类别解耦

来源：[DKD，CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html)。原文把目标类别与非目标类别关系分解；本轮图节点未标注，所以用固定教师argmax作为分组索引，不用query真标签，也不宣称严格复现原监督蒸馏协议。

设教师分布q、学生分布p，t是教师首选类别。普通KL可写为 $ KL(q||p)=KL([q_t,1-q_t]||[p_t,1-p_t])+(1-q_t)KL(q_{\neg t}/(1-q_t)||p_{\neg t}/(1-p_t)) $。数值实现对其他类别用logsumexp，不以减1近似导致精度损失。验证实际函数的恒等分解及对学生logits的梯度。

每种基础Prompt比较：原KL；DKD=二元KL+条件KL（两项系数1）；只有二元KL；只有条件KL；双倍原KL；无蒸馏。没有更换固定教师、门控、原型头、100步预算或主干；支撑LOO CE和原参数正则保持不变。借鉴机制而不照搬原文更大beta超参数，系数预先固定。

两种基础为修正原型教师gamma3与平滑岭后层gamma10，各6方法，外加Frozen、GraphPrompt、两种直接教师，共16×3图×2权重×5支持seed35100..35104=480条。保持同图同checkpoint同支持/query；先3步预检，强候选再固定新支持确认。无蒸馏通过实际loss置零实现，方法名中的kd不改变这个控制。
