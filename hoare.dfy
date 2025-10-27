method Min (x : int, y : int) returns (min : int)
ensures min <= x && min <= y;
ensures min == x || min == y; {
    if (x < y) {
        min := x;
    }
    else {
        min := y;
    }
}

method Max (x: int, y: int) returns (max: int) 
    ensures max >= x && max >= y;
    ensures max == x || max == y; {
        if (x < y) {
            max := y;
        } else {
            max := x;
        }
    }