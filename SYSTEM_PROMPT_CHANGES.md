# System Prompt Changes - Final Agent (Synthesis Phase)

## Overview
This document describes the updates made to the system prompt for the final synthesis agent in `pro_mode.py`.

## Previous System Prompt

```
You are an expert editor. You are given several answers from candidates. 
Your task is to review the answers and synthesize ONE best answer from the 
candidate answers provided by merging them, merging strengths, correcting errors, 
and removing repetition. Do not mention the candidates or the synthesis process. 
Be decisive and clear.
```

**Character count**: ~280 characters

## New System Prompt

```
You are an expert synthesizer and editor. Your role is to analyze multiple candidate 
answers and produce ONE superior final answer that represents the best collective intelligence.

Guidelines:
1. MERGE STRENGTHS: Identify the best insights, explanations, and examples from each candidate. 
Combine complementary information into a cohesive whole.
2. CORRECT ERRORS: Fix factual mistakes, logical inconsistencies, or misleading statements. 
If candidates disagree, use reasoning to determine the most accurate position.
3. ELIMINATE REDUNDANCY: Remove duplicate information and repetitive phrasing while preserving 
unique contributions from each candidate.
4. ENHANCE CLARITY: Reorganize and rewrite for maximum clarity, coherence, and readability. 
Use clear structure when appropriate (e.g., lists, sections).
5. BE COMPREHENSIVE YET CONCISE: Include all important information but express it efficiently. 
Avoid unnecessary verbosity.
6. MAINTAIN OBJECTIVITY: Present a balanced, well-reasoned answer. Do not mention the synthesis 
process, candidates, or your role as an editor.

Output ONLY the final synthesized answer - nothing else.
```

**Character count**: ~1,134 characters

## Key Improvements

### 1. **Enhanced Role Definition**
- **Before**: "expert editor"
- **After**: "expert synthesizer and editor" with emphasis on "collective intelligence"
- **Why**: More accurately describes the agent's purpose and establishes higher expectations

### 2. **Structured Guidelines**
- **Before**: Single paragraph with comma-separated instructions
- **After**: 6 numbered, explicit guidelines with clear action items
- **Why**: Easier for LLMs to parse and follow; reduces ambiguity

### 3. **Explicit Quality Control**
- **New**: "Fix factual mistakes, logical inconsistencies, or misleading statements"
- **New**: "If candidates disagree, use reasoning to determine the most accurate position"
- **Why**: Encourages critical thinking and fact-checking, not just text merging

### 4. **Clarity and Structure Guidance**
- **New**: "Reorganize and rewrite for maximum clarity, coherence, and readability"
- **New**: "Use clear structure when appropriate (e.g., lists, sections)"
- **Why**: Promotes better-organized, more readable final answers

### 5. **Balance Instruction**
- **New**: "BE COMPREHENSIVE YET CONCISE"
- **Why**: Addresses the common LLM tendency toward verbosity while ensuring completeness

### 6. **Clearer Output Directive**
- **Before**: Implicit in "do not mention the synthesis process"
- **After**: Explicit final line: "Output ONLY the final synthesized answer - nothing else"
- **Why**: Reduces meta-commentary and focuses output on the actual answer

## Expected Impact

1. **Higher Quality Synthesis**: More thorough analysis of candidates
2. **Better Error Correction**: Explicit instruction to fix mistakes and resolve disagreements
3. **Improved Readability**: Encouragement of structure and organization
4. **More Focused Output**: Clearer directive to avoid process discussion
5. **Better Balance**: Guidance on being comprehensive without verbosity

## Testing Recommendations

To validate these improvements, test with:

1. **Candidates with errors**: Ensure synthesis corrects factual mistakes
2. **Candidates with disagreements**: Verify reasoned resolution of conflicts
3. **Redundant candidates**: Check that repetition is eliminated
4. **Complex topics**: Assess structure and clarity improvements
5. **Verbose candidates**: Confirm conciseness while maintaining completeness
