from z3 import *

class Z3Verifier:
    @staticmethod
    def verify_equivalence(expr1, expr2, timeout_ms=5000):
        s = Solver()
        s.set("timeout", timeout_ms)
        s.add(expr1 != expr2)
        result = {"equivalent": False, "status": None}
        try:
            if s.check() == unsat:
                result["equivalent"] = True
                result["status"] = "unsat"
            else:
                result["status"] = "sat"
        except Exception as e:
            result["status"] = "error"
        return result
