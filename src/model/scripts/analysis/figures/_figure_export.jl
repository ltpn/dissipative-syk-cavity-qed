function pdf_hires_bbox(path::AbstractString)
    gs = Sys.which("gs")
    gs === nothing && error("PDF export requires Ghostscript (`gs`)")
    stderr_buffer = IOBuffer()
    run(pipeline(Cmd([gs, "-q", "-dNOPAUSE", "-dBATCH",
                      "-sDEVICE=bbox", path]);
                 stdout = devnull, stderr = stderr_buffer))
    output = String(take!(stderr_buffer))
    matched = match(
        r"%%HiResBoundingBox:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
        output)
    matched === nothing && error("Could not determine PDF bounding box for $path")
    return parse.(Float64, matched.captures)
end

function crop_pdf_hires(input_path::AbstractString,
                        output_path::AbstractString; margin_pt::Real = 0)
    llx, lly, urx, ury = pdf_hires_bbox(input_path)
    padded_llx = llx - margin_pt
    padded_lly = lly - margin_pt
    width = urx - llx + 2 * margin_pt
    height = ury - lly + 2 * margin_pt
    width > 0 && height > 0 || error("Invalid PDF bounding box for $input_path")

    pdfcrop = Sys.which("pdfcrop")
    if pdfcrop !== nothing
        run(Cmd([pdfcrop, "--hires", "--margins", string(margin_pt),
                 input_path, output_path]))
    else
        gs = Sys.which("gs")
        gs === nothing && error("PDF cropping requires Ghostscript (`gs`)")
        run(Cmd([gs, "-q", "-dNOPAUSE", "-dBATCH",
                 "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5", "-dFIXEDMEDIA",
                 "-dDEVICEWIDTHPOINTS=$(width)", "-dDEVICEHEIGHTPOINTS=$(height)",
                 "-sOutputFile=$(output_path)", "-c",
                 "<</PageOffset [-$(padded_llx) -$(padded_lly)]>> setpagedevice",
                 "-f", input_path]))
    end
    return [width, height]
end

function rasterize_pdf_png(pdf_path::AbstractString,
                           png_path::AbstractString)
    pdftocairo = Sys.which("pdftocairo")
    if pdftocairo !== nothing
        output_base = splitext(png_path)[1]
        run(Cmd([pdftocairo, "-png", "-singlefile", "-r", "288",
                 pdf_path, output_base]))
    else
        gs = Sys.which("gs")
        gs === nothing && error("PNG export requires Ghostscript (`gs`)")
        run(Cmd([gs, "-q", "-dNOPAUSE", "-dBATCH",
                 "-sDEVICE=pngalpha", "-r288", "-dTextAlphaBits=4",
                 "-dGraphicsAlphaBits=4", "-sOutputFile=$(png_path)",
                 pdf_path]))
    end
    return png_path
end
