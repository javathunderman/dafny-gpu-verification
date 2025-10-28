class ProgramState {
    const b_dim : int
    const g_dim : int
    var blocks : seq<seq<set<int>>>
    ghost predicate Valid() reads this { // not sure if there is some danger in saying it's spec-only/a ghost predicate
        forall i :: 0 <= i < |blocks| ==> |blocks[i]| == |blocks[0]|
    }

    constructor (b_dim: int, g_dim: int, acc: set<int>) ensures Valid() requires b_dim >= 0 requires g_dim >= 0 {
        this.b_dim := b_dim;
        this.g_dim := g_dim;
        blocks := seq (g_dim, i => seq(b_dim, j => acc));
    }

    lemma ValidMemAccess()
        requires Valid()
        ensures (forall b :: 0 <= b < |blocks| ==> 
                    (forall k :: 0 <= k < |blocks[b]| ==> 
                        (forall m :: m in blocks[b][k] && m >= 0 && m < b_dim)))
        {
        }
}
