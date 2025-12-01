class ProgramState {
    const b_dim : int
    const g_dim : int
    var blocks : seq<seq<set<int>>>
    predicate Valid() reads this {
        forall blockIdx :: 0 <= blockIdx < |blocks| ==> |blocks[blockIdx]| == |blocks[0]|
        && 
        forall blockIdx :: 0 <= blockIdx < |blocks| ==>
            forall threadIdx :: 0 <= threadIdx < |blocks[blockIdx]| ==>
                forall memAddr :: memAddr in blocks[blockIdx][threadIdx] ==>
                    memAddr > 0
                    &&
                    memAddr < |blocks[blockIdx]|
                    //true
                    // forall notBlockIdx :: 0 <= notBlockIdx < |blocks| && notBlockIdx != blockIdx ==>
                    //     forall notThreadIdx :: 0 <= notThreadIdx < |blocks[notBlockIdx]| && notThreadIdx != threadIdx ==>
                    //         forall notMemAddr :: notMemAddr in blocks[notBlockIdx][notThreadIdx] ==>
                    //             memAddr != notMemAddr
        // says all of the block sizes should be the same & 
        
    }

    constructor (b_dim: int, g_dim: int, acc: set<int>)
        ensures Valid()
        requires b_dim >= 0
        requires g_dim >= 0 
        requires forall x :: x in acc ==>
            x > 0
            && 
            x < b_dim
    {
        this.b_dim := b_dim;
        this.g_dim := g_dim;
        blocks := seq (g_dim, i => seq(b_dim, j => acc));
    }

    lemma ValidMemAccess()
        requires Valid()
        ensures (forall b :: 0 <= b < |blocks| ==> 
                    (forall k :: 0 <= k < |blocks[b]| ==> 
                        //(forall m :: m in blocks[b][k] && m >= 0 && m < b_dim ==> true)))      
                        (forall m :: m in blocks[b][k] ==> m>0 && m<|blocks[b]|)))      
        {
        }
}
