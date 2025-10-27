datatype Process = Agnes | Agatha | Germaine | Jack
datatype CState = Thinking | Hungry | Eating

class TicketSystem {
    var ticket: int
    var serving: int
    const P: set<Process>
    var cs: map<Process, CState>
    var t: map<Process, int>

    predicate Valid() reads this {
        cs.Keys == t.Keys == P && serving <= ticket &&
        (forall p :: p in P && cs[p] != Thinking ==> serving <= t[p] < ticket) &&
        (forall p, q :: p in P && q in P && p != q && cs[p] != Thinking && cs[q] != Thinking ==> t[p] != t[q]) &&
        (forall p :: p in P && cs[p] == Eating ==> t[p] == serving)
    }

    constructor (processes: set<Process>) ensures Valid() {
        P := processes;
        ticket := serving;
        cs := map p | p in processes :: Thinking;
        t := map p | p in processes :: 0;
    }

    method Request(p: Process)
        requires Valid() && p in P && cs[p] == Thinking
        modifies this
        ensures Valid() {
            t, ticket := t[p := ticket], ticket + 1;
            cs := cs[p := Hungry];
    }

    method Enter0(p: Process) 
        requires Valid() && p in P && cs[p] == Hungry && t[p] == serving
        modifies this
        ensures Valid() {
            cs := cs[p := Eating];
    }

    method Enter1(p: Process) 
        requires Valid() && p in P && cs[p] == Hungry && t[p] != serving
        modifies this
        ensures Valid() 
        {
        }

    method Leave(p: Process)
        requires Valid() && p in P && cs[p] == Eating
        modifies this
        ensures Valid() {
            serving := serving + 1;
            cs := cs[p := Thinking];
    }
    
    lemma MutualExclusion(p: Process, q: Process)
    requires Valid() && p in P && q in P
    requires cs[p] == Eating && cs[q] == Eating
    ensures p == q
    {
    }

    method Run(processes: set<Process>)
    requires processes != {}
    // ensures P == processes
    decreases * // omit the termination checks since this method has a loop that never ends
    {
        var ts := new TicketSystem(processes);
        while exists p :: p in ts.P && (ts.cs[p] == Hungry ==> ts.t[p] == ts.serving)
            invariant ts.Valid()
            decreases * {
                var p :| p in ts.P && (ts.cs[p] == Hungry ==> ts.t[p] == ts.serving);
                match ts.cs[p] {
                    case Thinking => ts.Request(p);
                    case Hungry => ts.Enter0(p);
                    case Eating => ts.Leave(p);
                }
        }
    }
}

    