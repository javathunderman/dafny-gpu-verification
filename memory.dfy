datatype Block = Block(ind: int, dim: int)
datatype Thread = Thread(ind: int, blk: Block)

class ProgramState {
    const blocks : set<Block>
    const threads: set<Thread>
    var acc: map<Thread, set<int>>

    ghost predicate Valid() reads this { // not sure if there is some danger in saying it's spec-only/a ghost predicate
        
        (forall p, q :: p in blocks && q in blocks && p.dim == q.dim)  && // blocks must have the same dimensions
        (forall p, q :: p in blocks && q in blocks && p.ind != q.ind) && // cannot have the same index
        (forall p, q :: p in threads && q in threads && p.ind != q.ind) && // cannot have threads with the same index
        (forall p :: p in threads && p.ind < p.blk.dim)
    }

    constructor (blks: set<Block>, th: set<Thread>, a: map<Thread, set<int>>) ensures Valid() {
        blocks := blks;
        threads := th;
        acc := a;
    }

    lemma ValidMemAccess(p: Thread)
        requires Valid() && p in threads
        ensures (forall k :: k in acc[p] && k < p.blk.dim && k >= 0)
        {
        }
}
