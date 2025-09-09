def generate_all_counting_problems():
    """Generate counting problems (without the conflicting extended block)."""
    problems = []
    # Single-step next: n -> n+1 (n=1..5) => targets 2..6
    for n in range(1, 6):
        target = n + 1
        details = (n, '->', None)
        problems.append((details, target, target))
    # Two-number sequence: n, n+1 -> n+2 (n=1..4) => targets 3..6
    for n in range(1, 5):
        target = n + 2
        details = (n, '->', n+1)
        problems.append((details, target, target))
    return problems

def generate_all_addition_problems():
    problems = []
    for a1 in range(1, 6):
        for a2 in range(1, 6):
            target = a1 + a2
            details = (a1, '+', a2)
            problems.append((details, target, target))
    return problems
