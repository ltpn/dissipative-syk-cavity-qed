using JLD2

function atomic_jld2(writer::Function, path::AbstractString)
    mkpath(dirname(path))
    tmp = path * ".tmp.$(getpid())"
    try
        JLD2.jldopen(tmp, "w") do file
            writer(file)
        end
        mv(tmp, path; force = true)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
    return path
end
