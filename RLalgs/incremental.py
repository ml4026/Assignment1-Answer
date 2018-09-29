def estimate(OldEstimate, StepSize, Target):
    NewEstimate = OldEstimate + StepSize * (Target - OldEstimate)
    return NewEstimate