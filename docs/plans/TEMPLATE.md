# Plan: [Feature Name]

**Date**: YYYY-MM-DD
**Author**: plan-drafter agent
**Status**: DRAFT / IN REVIEW / APPROVED

---

## 1. Objective & Scope

### What
[What are we building/changing?]

### Why
[Why is this needed?]

### Out of Scope
- [What we are explicitly NOT doing]

---

## 2. Architecture & Design Decisions

### Approach
[High-level description of the approach]

### Key Decisions
| Decision | Rationale | Alternatives Rejected |
|----------|-----------|----------------------|
| | | |

---

## 3. Implementation Details

### File Changes (in order)

#### File: `path/to/file.py`
- **Action**: Create / Modify
- **Changes**: [Description]
- **Depends on**: [Other files that must be changed first]

---

## 4. Code Snippets & Interface Contracts

### Interface Contracts
(The test-writer uses these to write tests CONCURRENTLY with implementation.
Be precise — ambiguous contracts lead to broken tests.)

| Function/Method | Signature | Returns | Raises |
|----------------|-----------|---------|--------|
| `create_user` | `(email: str, name: str) -> User` | `User` dataclass | `ValidationError`, `DuplicateError` |
| | | | |

### Key Code Snippets

```python
# Key implementation snippet with full type signatures
```

---

## 5. Testing Strategy

### Unit Tests
| Test Case | Function | Input | Expected Output | Edge Case? |
|-----------|----------|-------|----------------|------------|
| | | | | |

### Integration Tests
- [ ] [Workflow description with expected inputs and outputs]

### Edge Cases
- [ ] [Specific edge case and what should happen]

---

## 6. Evaluation Criteria

- [ ] [Acceptance criterion 1]
- [ ] [Acceptance criterion 2]

---

## 7. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| | | | |

### Rollback Strategy
[How to undo these changes if needed]

---

## Review History

This section is filled in by the orchestrator after each review round.
It preserves how the plan evolved — useful for understanding why decisions
were made and for the documenter when creating ADRs.

### Round 1 — Plan Review
- **Reviewer**: plan-reviewer
- **Status**: [APPROVED / NEEDS REVISION]
- **Key feedback**: [What the reviewer flagged]
- **Changes made**: [How the plan was revised in response]

### Round 2 — Plan Review (if needed)
- **Status**: [APPROVED / NEEDS REVISION]
- **Key feedback**: [...]
- **Changes made**: [...]

### Gap Analysis
- **Round 1 verdict**: [CLEARED / ISSUES FOUND]
- **Issues identified**: [What the gap-finder found]
- **Resolutions**: [How each issue was addressed]
- **Round 2 verdict (if needed)**: [CLEARED / ISSUES FOUND]

### Decisions That Changed During Review
[List any design decisions that were revised based on reviewer or gap-finder
feedback. These are candidates for ADRs that capture the "we initially
thought X but switched to Y because Z" narrative.]
