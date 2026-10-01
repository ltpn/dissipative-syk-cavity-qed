using SparseArrays: sparse
using Base.Threads
using Combinatorics

# This function generates all the possible strings with N numbers where N/2 are 0 and N/2 are 1
generate_vectors(N::Int) = [((v = zeros(Int, N); v[c] .= 1; v)) for c in combinations(1:N, div(N, 2))]

# This function generates all the possible strings with N numbers where k are 1 and N-k are 0
generate_vectors(N::Int, k::Int) = [((v = zeros(Int, N); v[c] .= 1; v)) for c in combinations(1:N, k)]

# This function counts the number of needed swaps to order the string j_1, j_2, l_1, ..., l_N/2
count_swaps(v) = sum(x[1] < x[2] && v[x[1]] > v[x[2]] for x in combinations(1:length(v), 2))

function Combinations_SYK4(N)

    # We produce all the possible combinations for the rank-2 tensor, in the form (j_1, j_2, k_1, k_2)
    all_Combinations = Iterators.product(1:N, 1:N, 1:N, 1:N)

    # We keep only the combinations in the form (j_1, j_2, k_1, k_2) such that j_1 < j_2 and k_1 < k_2
    Combinations = collect(Iterators.filter(x -> x[1] < x[2] && x[3] < x[4], all_Combinations))

    to_delete = []

    # We keep only one of the two combinations (j_1, j_2, k_1, k_2) and (k_1, k_2, j_1, j_2)
    # ignoring the combinations like (j_1, j_2, j_1, j_2)
    for combination in Combinations
        if combination[1:2] != combination[3:4] && !(combination in to_delete)
            list_hc = (combination[3], combination[4], combination[1], combination[2])
            push!(to_delete, list_hc)
        end
    end

    # We delete those combinations from the full list
    for item in to_delete
        deleteat!(Combinations, findfirst(==(item), Combinations))
    end

    return Combinations
end

function fermion_sign_SYK4(Combination, indices_ket, indices_states)

    # This function computes the sign coming from the computation of a single matrix element for the SYK4

    # we compute the indices of the 1 in the Fock strings which give a nonzero expectation value, and then
    # we compute the position in the list l_1, ..., l_N/2 of the k_1 and k_2 elements. This gives an overall
    # sign equal to (-1)^(pos_k_1 + pos_k_2 - 2), since k2 > k1
    indices = map(index -> findfirst(x -> x == Combination[3], index), indices_states[indices_ket]) + map(index -> findfirst(x -> x == Combination[4], index), indices_states[indices_ket]) .- 2

    # we discard the elements k_1 and k_2 from the string l_1, ..., l_N/2
    indices_k1k2 = map(v -> filter(x -> x != Combination[3] && x != Combination[4], v), indices_states[indices_ket])
    # we concatenate the elements j_1 and j_2 at the beginning of the string l_1, ..., l_N/2
    indices_j1j2 = map(v -> vcat([Combination[1], Combination[2]], v), indices_k1k2)

    # we count the number of swaps needed to order the string j_1, j_2, l_1, ..., l_N/2. This gives an
    # overall sign equal to (-1)^(n_swaps)
    swap_counts = map(count_swaps, indices_j1j2)

    # the sign coming from the manipulation of the fermionic operators thus reads (-1).^(n_swaps .+ pos_k_1 + pos_k_2 - 2)
    return (-1) .^ (swap_counts .+ indices)
end

function SYK4_block_direct(N, basis_states, J_list)

    # We generate the list of the indices where a 1 appears in the Fock strings of basis_states
    indices_states = [findall(x -> x == 1, state) for state in basis_states]

    # We construct the half-filling matrix of the SYK(q) model
    H_SYK = zeros(ComplexF64, size(basis_states)[1], size(basis_states)[1])

    # We produce all the relevant combinations (j_1, j_2, k_1, k_2) in the SYK(4) sum
    Combinations = Combinations_SYK4(N)

    # Loop over the Hamiltonian terms
    for (j, Combination) in enumerate(Combinations)

        # Given the combination (j_1, j_2, k_1, k_2), we construct the reduced basis for which we have a nonzero result
        red_basis = collect(Iterators.filter(x -> x[Combinations[j][3]] == 1 && x[Combinations[j][4]] == 1, basis_states))
        red_basis = collect(Iterators.filter(x -> (x[Combinations[j][1]] == 0 || Combinations[j][1] == Combinations[j][3] || Combinations[j][1] == Combinations[j][4]) && (x[Combinations[j][2]] == 0 || Combinations[j][2] == Combinations[j][3] || Combinations[j][2] == Combinations[j][4]), red_basis))

        # We copy the results i the ket basis
        red_basis_ket = [copy(x) for x in red_basis]

        # We compute the list of the indices of the relvant kets among all the basis states
        indices_ket = findall(x -> x in red_basis_ket, basis_states)

        # We compute the fermions signs for the list of the relevant ket basis
        sign = fermion_sign_SYK4(Combination, indices_ket, indices_states)

        # We act with the Hamiltonian element identified by (j_1, j_2, k_1, k_2), obtaining the bra basis
        foreach(x -> (x[Combinations[j][3]] -= 1; x[Combinations[j][4]] -= 1), red_basis_ket)
        foreach(x -> (x[Combinations[j][1]] += 1; x[Combinations[j][2]] += 1), red_basis_ket)
        red_basis_bra = red_basis_ket

        # We compute the list of the indices of the relvant bras among all the basis states
        indices_bra = findall(x -> x in red_basis_bra, basis_states)

        # We fill the SYK4 block at half filling for the combination (j_1, j_2, k_1, k_2)
        H_SYK[CartesianIndex.(indices_bra, indices_ket)] .+= 4 * sign * J_list[j]
        if Combination[1:2] != Combination[3:4]
            H_SYK[CartesianIndex.(indices_ket, indices_bra)] .+= 4 * sign * J_list[j]'
        end
    end

    return sparse(H_SYK)
end

############## SYK LIOUVILLIAN #################################

function fermion_sign_SYK2(Combination, indices_ket, indices_states)

    # This function computes the sign coming from the computation of a single matrix element for the SYK2 dissipator term c_j' c_k

    # we compute the position in the list l_1, ..., l_N/2 of the k element.
    pos_k = map(index -> findfirst(x -> x == Combination[2], index), indices_states[indices_ket])

    # we discard the element k from the string l_1, ..., l_N/2
    indices_k = map(v -> filter(x -> x != Combination[2], v), indices_states[indices_ket])
    # we concatenate the element j at the beginning of the string l_1, ..., l_N/2
    indices_j = map(v -> vcat([Combination[1]], v), indices_k)

    # we count the number of swaps needed to order the string j, l_1, ..., l_N/2. This gives an
    # overall sign equal to (-1)^(n_swaps)
    swap_counts = map(count_swaps, indices_j)

    # the sign coming from the manipulation of the fermionic operators thus reads (-1).^(n_swaps + pos_k -1)
    return (-1) .^ (swap_counts .+ pos_k .- 1)
end

function dissipator_block_direct(N, basis_states, K_list)

    # We generate the list of the indices where a 1 appears in the Fock strings of basis_states
    indices_states = [findall(x -> x == 1, state) for state in basis_states]

    # We construct the half-filling matrix of the SYK(q) model
    L_SYK = zeros(ComplexF64, size(basis_states)[1], size(basis_states)[1])

    # We produce all the relevant combinations (j, k) in the SYK(2) sum
    Combinations = vec(collect(Iterators.product(1:N, 1:N)))

    # Loop over the dissipator terms
    for (j, Combination) in enumerate(Combinations)

        # Given the combination (j, k), we construct the reduced basis for which we have a nonzero result
        # k must be in the state (particle), j must not be in the state (hole), unless j=k
        red_basis = collect(Iterators.filter(x -> x[Combination[2]] == 1 && (x[Combination[1]] == 0 || Combination[1] == Combination[2]), basis_states))

        if isempty(red_basis)
            continue
        end

        # We copy the results in the ket basis
        red_basis_ket = [copy(x) for x in red_basis]

        # We compute the list of the indices of the relevant kets among all the basis states
        indices_ket = findall(x -> x in red_basis_ket, basis_states)

        # We compute the fermions signs for the list of the relevant ket basis
        sign = fermion_sign_SYK2(Combination, indices_ket, indices_states)

        # We act with the dissipator element identified by (j, k), obtaining the bra basis
        foreach(x -> (x[Combination[2]] -= 1), red_basis_ket)
        foreach(x -> (x[Combination[1]] += 1), red_basis_ket)
        red_basis_bra = red_basis_ket

        # We compute the list of the indices of the relevant bras among all the basis states
        indices_bra = findall(x -> x in red_basis_bra, basis_states)

        # We fill the SYK dissipator block at half filling for the combination (j, k)
        L_SYK[CartesianIndex.(indices_bra, indices_ket)] .+= sign .* K_list[j]
    end

    return sparse(L_SYK)
end

#######
