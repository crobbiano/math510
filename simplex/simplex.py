# Simplex algorithm for MATH510 HW1 / Midterm
# This algorithm finds an optimal solution given it exists
# for the problem of:
#       min Ax=b given some constraint
#
# Steps:
#   1 - Initialize tableux, T, with data. T(1,1) is total cost,
#       T(1, 2:end) is relative costs, T(2:end, 1) is inv(ARREF)*b
#       and is the x that solves the current A matrix,
#       T(2:end, 2:end) is the A matrix (m by n, m<n).
#   2 - Find an initial bfs by RREF the A matrix, while keeping
#       track of the relative costs and the inv(ARREF)*b column
#   3 - Clear the relative costs row in the bfs columns
#   4 - Choose an entering column by picking the largest negative
#       relative cost column to enter
#   5 - Choose an exiting column by the theta rule, then scale the
#       entering column by the best ratio found with the theta rule
#   6 - Continue steps 2-5 until no relative cost is negative
import sys, getopt
import numpy as np
import sympy as sp

def like_a_gauss(mat, b):
    """
    Changes mat into Reduced Row-Echelon Form.
    """
    # Let's do forward step first.
    # at the end of this for loop, the matrix is in Row-Echelon format.
    for i in range(min(len(mat), len(mat[0]))):
        # every iteration, ignore one more row and column
        for r in range(i, len(mat)):
            # find the first row with a nonzero entry in first column
            zero_row = mat[r][i] == 0
            if zero_row:
                continue
            # swap current row with first row
            mat[i], mat[r] = mat[r], mat[i]
            # swap the current row with first row in the vector
            b[i], b[r] = b[r], b[i]
            # add multiples of the new first row to lower rows so lower
            # entries of first column is zero
            first_row_first_col = mat[i][i]
            for rr in range(i + 1, len(mat)):
                this_row_first = mat[rr][i]
                scalarMultiple = -1 * this_row_first / first_row_first_col
                for cc in range(i, len(mat[0])):
                    mat[rr][cc] += mat[i][cc] * scalarMultiple
            break

    # At the end of the forward step
    # print(mat)
    # Now reduce
    for i in range(min(len(mat), len(mat[0])) - 1, -1, -1):
        # divide last non-zero row by first non-zero entry
        first_elem_col = -1
        first_elem = -1
        for c in range(len(mat[0])):
            if mat[i][c] == 0:
                continue
            if first_elem_col == -1:
                first_elem_col = c
                first_elem = mat[i][c]
            mat[i][c] /= first_elem
        # add multiples of this row so all numbers above the leading 1 is zero
        for r in range(i):
            this_row_above = mat[r][first_elem_col]
            scalarMultiple = -1 * this_row_above
            for cc in range(len(mat[0])):
                mat[r][cc] += mat[i][cc] * scalarMultiple
        # disregard this row and continue
    # print(mat)

def simplex(A, b, relcost, totalcost):
    # Do the steps here
    # This is the initial RREF step
    print('Found it', A, ' and ', b, ' and ', relcost, ' and ', totalcost)
    m = A.shape[0]
    n = A.shape[1]
    identity = np.identity(m)
    concat = np.concatenate((A, identity), axis=1)
    # print(concat)
    # print(b)
    # RREF the concatenated matrix
    like_a_gauss(concat, b)
    Binv = concat[:,n:]
    ARREF = concat[:,0:n]
    print('ARREF: ', ARREF)
    # print(ARREF.shape)
    # print(Binv)
    # print(b)
    b = Binv.dot(b)
    print('b', b)

    # Capture the indices of current BFS
    BFSidx=np.array(range(m))
    # print('BFSidx: ', BFSidx)

    # This is where we clear the relcost above pivots and update total cost
    # print(relcost, relcost[0],ARREF[0,:])
    for idx in range(0,m):
        totalcost += -relcost[idx]*b[idx]
        relcost += -relcost[idx]*ARREF[idx,:]
    # print('costs1: ', relcost, totalcost)

    loopIdx = 0;
    while(1):
        loopIdx+=1
        print('\n\nLoop: ', loopIdx, ' RelCosts: ', relcost)
        print('BFSidx: ', BFSidx)
        # Find the column indexes where the relative costs is < 0
        nonBFSidx = list(set(range(0,n)) - set(BFSidx))
        # T = [relcost[i] for i in nonBFSidx]
        # print('T',T)
        negCosts = np.array([i for i,v in enumerate(relcost) if v < -0])
        # print('negCosts: ',negCosts)
        # negCosts = [i for i,v in enumerate(T) if v < 0]
        # print('relcost: ', relcost)
        # Find the largest negative relative cost and swap in that column
        # sortedNegCosts = negCosts.argsort()[:]
        sortedNegCosts = relcost.argsort()[:len(negCosts)]
        negCostIdx = 0
        print('Sorted negCosts: ',sortedNegCosts)
        enteringColumn = sortedNegCosts[negCostIdx]
        print('Entering: ', enteringColumn, ' ', ARREF[:,enteringColumn])
        theta = np.divide(b,ARREF[:,enteringColumn])
        print('Theta: ', theta)

        while (all(theta <= 0) and negCostIdx<len(negCosts)):
            print('while idx: ', negCostIdx)
            enteringColumn = negCosts[negCostIdx]
            print('Entering: ', enteringColumn, ' ', ARREF[:,enteringColumn])
            theta = np.divide(b,ARREF[:,enteringColumn])
            print('Theta: ', theta)
            negCostIdx+=1


        # Find the exiting column via the theta rule
        value,position = min(((c,a) for a,c in enumerate(theta) if c>0), default=(None,None))
        print(position, value)
        # exitingColumn = m + (position+1)
        exitingColumn = BFSidx[position]
        print('Exiting: ', exitingColumn)

        concat = np.concatenate((ARREF, identity), axis=1)
        # print(concat)
        # print(ARREF)
        # Clear rows above and below the new pivot
        # Normalize the pivot row
        # print(concat[exitingColumn,:])
        # print('concat: ', concat)
        rowIdx, = np.where(BFSidx==exitingColumn);
        print('rowIdx',rowIdx)
        concat[rowIdx,:] /= concat[rowIdx, enteringColumn]
        # print('concat: ', concat[BFSidx[BFSidx==exitingColumn],:])
        for idx in range(0,m):
            if (idx == rowIdx):
                # skip
                print('Skipping row: ', idx)
            else:
                # row operation to clear rows
                print('rowScaled',-concat[idx, enteringColumn]*concat[rowIdx,:])
                print('rowScaled',(-concat[idx, enteringColumn]*concat[rowIdx,:]).shape)
                print('rowScaled',(-concat[idx, enteringColumn]*concat[exitingColumn,:]).shape)
                print('rowScaled',(concat[idx,:]).shape)
                concat[idx,:] += -concat[idx, enteringColumn]*concat[rowIdx,:]
                # print('after', concat[idx,:] )
        print('after: ',concat)

        # Update the BFSidx
        BFSidx[exitingColumn] = enteringColumn
        # print('BFSidx: ', BFSidx)


        # FIXME something below here is broken - the indexing of everything is whacked
        Binvsingle = concat[:,n:]
        Binv = Binv*Binvsingle
        ARREF = concat[:,0:n]
        print('Binv: ', Binv)
        # print(ARREF)
        b = Binv.dot(b)
        print('b: ',b)
        for idx in range(0,m):
            totalcost += -relcost[BFSidx[BFSidx==enteringColumn]]*b[idx]
            relcost += -relcost[BFSidx[BFSidx==enteringColumn]]*ARREF[idx,:]
        print('costs: ', relcost, totalcost)
        if (all(relcost>=0)):
            break

    print('Finished')
    print('costs: ', relcost, totalcost)

if __name__ == '__main__':
    # Handle input args FIXME - doesn't error on no input
    argv=sys.argv[1:]
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    # print('Input file is ', inputfile)

    # Read in CSV file with all data and save
    tableux = np.genfromtxt(inputfile, delimiter=',')
    # Form the A, b and cost vectors
    relcost = tableux[0, 1:]
    totalcost = tableux[0,0]
    A = tableux[1:,1:]
    b = tableux[1:,0]

    # Read in the csv file passed in on the command line
    simplex(A, b, relcost, totalcost)
