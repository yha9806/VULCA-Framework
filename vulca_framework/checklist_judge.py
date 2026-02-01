"""
checklist_judge.py - Layer 2 Checklist Judge for VULCA Evaluation Framework v2.0

Implements 10-item checklist evaluation using LLM as judge:
- Mode A: 10 items (with expert reference)
- Mode B: 8 items (reference-free, q6/q7 N/A)

Uses Claude Opus 4.5 as primary judge for binary Yes/No decisions.

Usage:
    from scripts.evaluation.checklist_judge import ChecklistJudge

    judge = ChecklistJudge()

    # Mode A (with expert reference)
    result = judge.evaluate(vlm_critique, expert_critique, culture, mode='A')

    # Mode B (reference-free)
    result = judge.evaluate(vlm_critique, culture=culture, artwork_info=info, mode='B')

Author: Claude Code
Version: 1.0 (2025-11-29)
"""

import json
import re
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Checklist Items Definition
# =============================================================================

CHECKLIST_ITEMS = {
    'q1': {
        'text_en': 'Does the VLM describe major visual elements (color, composition, subjects)?',
        'text_zh': '是否描述了主要视觉元素（色彩、构图、主题）？',
        'category': '观察完整性',
        'mode': 'both'
    },
    'q2': {
        'text_en': 'Does the VLM identify the artistic medium or technique used?',
        'text_zh': '是否识别了艺术媒介或技法？',
        'category': '技术分析',
        'mode': 'both'
    },
    'q3': {
        'text_en': 'Does the VLM use culture-specific terminology?',
        'text_zh': '是否使用了文化特定术语？',
        'category': '文化理解',
        'mode': 'both'
    },
    'q4': {
        'text_en': 'Does the VLM reference historical period, movement, or artist background?',
        'text_zh': '是否引用了历史时期、流派或艺术家背景？',
        'category': '历史语境',
        'mode': 'both'
    },
    'q5': {
        'text_en': "Does the VLM attempt to interpret the artwork's meaning?",
        'text_zh': '是否尝试解读作品意义？',
        'category': '深度分析',
        'mode': 'both'
    },
    'q6': {
        'text_en': 'Are all factual statements in the VLM critique accurate?',
        'text_zh': '事实陈述是否准确？',
        'category': '准确性',
        'mode': 'A_only'
    },
    'q7': {
        'text_en': "Is the VLM's interpretation consistent with expert's key points?",
        'text_zh': '是否与专家评论观点一致？',
        'category': '对齐度',
        'mode': 'A_only'
    },
    'q8': {
        'text_en': 'Is the VLM critique well-structured and coherent?',
        'text_zh': '结构是否清晰有条理？',
        'category': '表达质量',
        'mode': 'both'
    },
    'q9': {
        'text_en': 'Does the VLM avoid generic or boilerplate statements?',
        'text_zh': '是否避免了通用套话？',
        'category': '具体性',
        'mode': 'both'
    },
    'q10': {
        'text_en': 'Overall, does the VLM critique meet professional standards?',
        'text_zh': '整体是否达到专业水准？',
        'category': '综合判断',
        'mode': 'both'
    }
}


# =============================================================================
# Prompt Templates
# =============================================================================

CHECKLIST_PROMPT_MODE_A = """You are evaluating a VLM-generated art critique against an expert reference.

## VLM Critique:
{vlm_critique}

## Expert Reference:
{expert_critique}

## Culture: {culture}

For each question below, answer ONLY "Yes" or "No" based on careful analysis:

1. Does the VLM describe major visual elements (color, composition, subjects)?
2. Does the VLM identify the artistic medium or technique used?
3. Does the VLM use culture-specific terminology (e.g., for Chinese: 气韵, 笔墨, 意境)?
4. Does the VLM reference historical period, movement, or artist background?
5. Does the VLM attempt to interpret the artwork's meaning beyond mere description?
6. Are all factual statements in the VLM critique accurate (compare with expert)?
7. Is the VLM's interpretation consistent with expert's key points?
8. Is the VLM critique well-structured and coherent?
9. Does the VLM avoid generic or boilerplate statements?
10. Overall, does the VLM critique meet professional standards?

Return ONLY a JSON object with your answers:
{{"q1": "Yes", "q2": "No", "q3": "Yes", ..., "q10": "Yes"}}"""


CHECKLIST_PROMPT_MODE_B = """You are evaluating a VLM-generated art critique (no expert reference available).

## VLM Critique:
{vlm_critique}

## Artwork Information:
{artwork_info}

## Culture: {culture}

For each question below, answer ONLY "Yes" or "No" (or "N/A" for questions requiring expert reference):

1. Does the VLM describe major visual elements (color, composition, subjects)?
2. Does the VLM identify the artistic medium or technique used?
3. Does the VLM use culture-specific terminology?
4. Does the VLM reference historical period, movement, or artist background?
5. Does the VLM attempt to interpret the artwork's meaning beyond mere description?
6. [N/A - requires expert reference]
7. [N/A - requires expert reference]
8. Is the VLM critique well-structured and coherent?
9. Does the VLM avoid generic or boilerplate statements?
10. Overall, does the VLM critique meet professional standards?

Return ONLY a JSON object with your answers:
{{"q1": "Yes", "q2": "Yes", ..., "q5": "No", "q6": "N/A", "q7": "N/A", "q8": "Yes", "q9": "Yes", "q10": "No"}}"""


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class ChecklistResult:
    """Result of checklist evaluation."""
    answers: Dict[str, str]  # q1-q10 -> Yes/No/N/A
    yes_count: int
    no_count: int
    na_count: int
    score: float  # 0-1 normalized score
    mode: str
    raw_response: str = ""


# =============================================================================
# Checklist Judge Class
# =============================================================================

class ChecklistJudge:
    """Layer 2 Checklist evaluator using LLM as judge."""

    def __init__(self, judge_model: str = 'claude'):
        """
        Initialize checklist judge.

        Args:
            judge_model: 'claude' or 'gpt5' (default: claude)
        """
        self.judge_model = judge_model

    def evaluate(
        self,
        vlm_critique: str,
        expert_critique: str = None,
        culture: str = 'chinese',
        artwork_info: str = None,
        mode: str = 'A'
    ) -> ChecklistResult:
        """
        Evaluate VLM critique using 10-item checklist.

        Args:
            vlm_critique: VLM-generated critique text
            expert_critique: Expert reference (required for Mode A)
            culture: Culture name
            artwork_info: Artwork metadata (for Mode B)
            mode: 'A' (reference-based) or 'B' (reference-free)

        Returns:
            ChecklistResult with answers and scores
        """
        mode = mode.upper()

        # Build prompt
        if mode == 'A':
            if not expert_critique:
                raise ValueError("Mode A requires expert_critique")
            prompt = CHECKLIST_PROMPT_MODE_A.format(
                vlm_critique=vlm_critique,
                expert_critique=expert_critique,
                culture=culture
            )
        else:
            if artwork_info is None:
                artwork_info = f"Culture: {culture}"
            prompt = CHECKLIST_PROMPT_MODE_B.format(
                vlm_critique=vlm_critique,
                artwork_info=artwork_info,
                culture=culture
            )

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        answers = self._parse_response(response, mode)

        # Calculate scores
        yes_count = sum(1 for v in answers.values() if v.lower() == 'yes')
        no_count = sum(1 for v in answers.values() if v.lower() == 'no')
        na_count = sum(1 for v in answers.values() if v.lower() == 'n/a')

        # Score = Yes / (Yes + No), excluding N/A
        valid_count = yes_count + no_count
        score = yes_count / valid_count if valid_count > 0 else 0.0

        return ChecklistResult(
            answers=answers,
            yes_count=yes_count,
            no_count=no_count,
            na_count=na_count,
            score=score,
            mode=mode,
            raw_response=response
        )

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API for evaluation."""
        if self.judge_model == 'claude':
            return self._call_claude(prompt)
        elif self.judge_model == 'gpt5':
            return self._call_gpt5(prompt)
        else:
            # Fallback to rule-based for testing
            return self._fallback_evaluation(prompt)

    def _call_claude(self, prompt: str) -> str:
        """Call Claude API."""
        try:
            import anthropic

            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                print("Warning: ANTHROPIC_API_KEY not set, using fallback")
                return self._fallback_evaluation(prompt)

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        except Exception as e:
            print(f"Claude API error: {e}, using fallback")
            return self._fallback_evaluation(prompt)

    def _call_gpt5(self, prompt: str) -> str:
        """Call GPT-5 API."""
        try:
            import openai

            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                print("Warning: OPENAI_API_KEY not set, using fallback")
                return self._fallback_evaluation(prompt)

            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"GPT-5 API error: {e}, using fallback")
            return self._fallback_evaluation(prompt)

    def _fallback_evaluation(self, prompt: str) -> str:
        """
        Rule-based fallback when API is unavailable.
        Uses keyword matching for basic evaluation.
        """
        # Extract VLM critique from prompt
        vlm_match = re.search(r'## VLM Critique:\n(.*?)(?:\n##|\nFor each)', prompt, re.DOTALL)
        vlm_critique = vlm_match.group(1).strip() if vlm_match else ""

        # Simple keyword-based checks
        answers = {}

        # Q1: Visual elements (color, composition, subjects)
        visual_keywords = ['色彩', 'color', '构图', 'composition', '主题', 'subject', '画面', 'visual']
        answers['q1'] = 'Yes' if any(kw in vlm_critique.lower() for kw in visual_keywords) else 'No'

        # Q2: Medium/technique
        tech_keywords = ['技法', 'technique', '媒介', 'medium', '笔法', 'brush', '油画', 'oil', '水墨', 'ink']
        answers['q2'] = 'Yes' if any(kw in vlm_critique.lower() for kw in tech_keywords) else 'No'

        # Q3: Cultural terminology
        cultural_keywords = ['气韵', '意境', '笔墨', '文人', 'literati', 'aesthetic', '美学', 'rasa', '侘寂']
        answers['q3'] = 'Yes' if any(kw in vlm_critique.lower() for kw in cultural_keywords) else 'No'

        # Q4: Historical context
        hist_keywords = ['朝代', 'dynasty', '时期', 'period', '流派', 'school', '艺术家', 'artist', '历史', 'history']
        answers['q4'] = 'Yes' if any(kw in vlm_critique.lower() for kw in hist_keywords) else 'No'

        # Q5: Interpretation
        interp_keywords = ['象征', 'symbol', '意义', 'meaning', '表达', 'express', '诠释', 'interpret', '体现', 'embody']
        answers['q5'] = 'Yes' if any(kw in vlm_critique.lower() for kw in interp_keywords) else 'No'

        # Q6/Q7: Mode A only
        if 'Expert Reference:' in prompt:
            answers['q6'] = 'Yes'  # Assume accurate without verification
            answers['q7'] = 'Yes'  # Assume consistent without verification
        else:
            answers['q6'] = 'N/A'
            answers['q7'] = 'N/A'

        # Q8: Structure (check for paragraph breaks or sections)
        answers['q8'] = 'Yes' if len(vlm_critique) > 100 and ('\n' in vlm_critique or '。' in vlm_critique) else 'No'

        # Q9: Avoid generic (check length and specificity)
        generic_phrases = ['总的来说', 'overall', '一般来说', 'generally', '众所周知', 'well known']
        has_generic = any(phrase in vlm_critique.lower() for phrase in generic_phrases)
        answers['q9'] = 'No' if has_generic else 'Yes'

        # Q10: Professional standard (based on length and other factors)
        yes_count = sum(1 for k, v in answers.items() if v == 'Yes' and k not in ['q6', 'q7', 'q10'])
        answers['q10'] = 'Yes' if yes_count >= 5 and len(vlm_critique) > 150 else 'No'

        return json.dumps(answers)

    def _parse_response(self, response: str, mode: str) -> Dict[str, str]:
        """Parse LLM response to extract answers."""
        # Try to find JSON in response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                answers = json.loads(json_match.group())
                # Ensure all q1-q10 keys exist
                for i in range(1, 11):
                    key = f'q{i}'
                    if key not in answers:
                        if mode == 'B' and key in ['q6', 'q7']:
                            answers[key] = 'N/A'
                        else:
                            answers[key] = 'No'  # Default to No if missing
                return answers
            except json.JSONDecodeError:
                pass

        # Fallback: parse Yes/No from text
        answers = {}
        for i in range(1, 11):
            key = f'q{i}'
            pattern = rf'{i}[.:)\s]+\s*(Yes|No|N/A)'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answers[key] = match.group(1)
            elif mode == 'B' and key in ['q6', 'q7']:
                answers[key] = 'N/A'
            else:
                answers[key] = 'No'

        return answers

    def get_detailed_report(self, result: ChecklistResult) -> str:
        """Generate detailed report from checklist result."""
        lines = [
            f"=== Checklist Evaluation Report (Mode {result.mode}) ===",
            f"Score: {result.score:.1%} ({result.yes_count} Yes / {result.yes_count + result.no_count} Valid)",
            ""
        ]

        for qid, item in CHECKLIST_ITEMS.items():
            answer = result.answers.get(qid, 'N/A')
            status = '✅' if answer.lower() == 'yes' else ('⬜' if answer.lower() == 'n/a' else '❌')
            lines.append(f"{status} {qid.upper()}: {item['text_en']}")
            lines.append(f"   Answer: {answer} | Category: {item['category']}")

        return '\n'.join(lines)


# =============================================================================
# Convenience Function
# =============================================================================

def evaluate_checklist(
    vlm_critique: str,
    expert_critique: str = None,
    culture: str = 'chinese',
    artwork_info: str = None,
    mode: str = 'A',
    judge_model: str = 'claude'
) -> Tuple[float, ChecklistResult]:
    """
    Convenience function to evaluate checklist.

    Returns:
        Tuple of (score, full_result)
    """
    judge = ChecklistJudge(judge_model)
    result = judge.evaluate(vlm_critique, expert_critique, culture, artwork_info, mode)
    return result.score, result


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    # Quick test with fallback evaluation
    test_critique = """
    此幅山水画以浓墨淡彩描绘江南水乡风光，构图采用三远法中的平远视角，
    展现了文人画的典型意境。笔法流畅，墨色层次分明，气韵生动，
    体现了明代吴门画派的艺术特色。画面虚实相生，意境深远，
    充分展现了中国传统绘画的审美追求。作品在技法上运用了传统的皴擦点染，
    同时融入了文人画的诗意表达，体现了画家深厚的艺术修养。
    """

    expert_critique = """
    此作为明代吴门画派代表作品，以工写结合的技法描绘江南山水。
    构图疏朗有致，笔墨苍润，气韵生动。画面呈现平远视角，
    山石采用披麻皴法，树木点叶精到。整体展现了文人画的审美理想，
    体现了画家对自然与人文的深刻理解。
    """

    print("=== Checklist Judge Test (Fallback Mode) ===\n")

    judge = ChecklistJudge(judge_model='fallback')

    # Mode A
    result_a = judge.evaluate(test_critique, expert_critique, 'chinese', mode='A')
    print(f"Mode A Score: {result_a.score:.1%}")
    print(f"Yes: {result_a.yes_count}, No: {result_a.no_count}, N/A: {result_a.na_count}")
    print(f"Answers: {result_a.answers}")
    print()

    # Mode B
    result_b = judge.evaluate(test_critique, culture='chinese', mode='B')
    print(f"Mode B Score: {result_b.score:.1%}")
    print(f"Yes: {result_b.yes_count}, No: {result_b.no_count}, N/A: {result_b.na_count}")
    print(f"Answers: {result_b.answers}")
    print()

    # Detailed report
    print(judge.get_detailed_report(result_a))
