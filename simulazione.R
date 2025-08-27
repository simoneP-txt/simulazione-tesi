# SETTING INIZIALE --------------------------------------------------------
rm(list = ls(all.names = TRUE), envir = .GlobalEnv)
gc()

pkgs <- c("data.table", "clusterGeneration", "mlbench", "igraph", "sbm",
          "RSpectra", "reticulate", "FNN", "tictoc", "irlba", "Matrix",
          "future.apply", "plotly", "tidyverse", "ggplot2", "pracma",
          "alphashape3d", "rpart", "mclust", "ntfy", "here", "beepr")

to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) {
  message("Installo pacchetti mancanti: ", paste(to_install, collapse = ", "))
  install.packages(to_install, dependencies = TRUE)
}

invisible(lapply(pkgs, library, character.only = TRUE))
library(here)

PY311 <- "C:/Users/simop/AppData/Local/Programs/Python/Python311/python.exe"

# processo principale
options(reticulate.python = PY311)
Sys.setenv(RETICULATE_PYTHON = PY311)

Sys.setenv(R_MAX_VSIZE = "14Gb",
           OMP_NUM_THREADS = parallel::detectCores())  # sklearn/umap

future::plan(future::multisession, workers = parallel::detectCores())

# FUNZIONI DI SUPPORTO ----------------------------------------------------
## Matrice AR(1) di correlazione
ar1_cor <- function(d, rho = 0.5) {
  exponent <- abs(
    matrix(1:d - 1, nrow = d, ncol = d, byrow = TRUE) -
      (1:d - 1)
  )
  rho^exponent
}

run_pca_2d <- function(X, k = 2) {
  sv <- RSpectra::svds(X, k = k, center = TRUE, scale = FALSE)
  sv$u %*% diag(sv$d)              # matrice n × k delle prime k componenti
}

shift_fun <- function(d, livello = c("facile", "medio", "difficile")) {
  livello <- match.arg(livello)
  base <- switch(livello,
                 "facile"    = 5.0,  # Distanza totale = 5.0
                 "medio"     = 2.5,   # Distanza totale = 2.5
                 "difficile" = 2.0)   # Distanza totale = 1.0
  per_coord <- base / sqrt(d)
  rep(per_coord, d)
}

simulate_once <- function(n, d,
                          rho     = 0.5,
                          livello = "medio") {
  
  shift_vec <- shift_fun(d, livello)
  
  y  <- sample(0:1, n, replace = TRUE)
  n1 <- sum(y == 0); n2 <- n - n1
  
  mu1  <- rep(0, d)
  mu2  <- shift_vec                    # media traslata
  Sig1 <- diag(d)                     # I_d
  Sig2 <- ar1_cor(d, rho)             # AR(1) con parametro rho
  
  # generazione dei campioni
  X1 <- MASS::mvrnorm(n1, mu1, Sig1)  # :contentReference[oaicite:1]{index=1}
  X2 <- MASS::mvrnorm(n2, mu2, Sig2)  # :contentReference[oaicite:2]{index=2}
  
  X <- matrix(NA_real_, nrow = n, ncol = d)
  X[y == 0, ] <- X1
  X[y == 1, ] <- X2
  
  list(X = X, y = y)
}

library(plotly)
library(RSpectra)   # per la PCA veloce con SVD

plot_simulation_3d <- function(sim,
                               titolo         = "Simulazione",
                               forza_pca      = TRUE,    
                               colori         = c("#1f77b4", "#ff7f0e"),
                               marker_size    = 2,
                               opacity_points = 0.7) {
  
  X  <- sim$X
  d  <- ncol(X)
  
  ## 1. Eventuale riduzione a 3 dimensioni con PCA
  if (forza_pca || d > 3) {
    # centre = TRUE per togliere la media; scale = FALSE per non normalizzare
    sv   <- RSpectra::svds(X, k = 3, center = TRUE, scale = FALSE)
    Z    <- sv$u %*% diag(sv$d)              # score n × 3
    axis <- paste0("PC", 1:3)
  } else {
    Z    <- X[, 1:min(3, d)]
    axis <- paste0("Dim ", 1:ncol(Z))
    # se d == 2 aggiungiamo una terza colonna di zeri per la Z-axis
    if (ncol(Z) == 2) {
      Z <- cbind(Z, Z3 = 0)
      axis <- c(axis, "Dim 3 (0)")
    }
  }
  
  ## 2. Plot interattivo
  idx0 <- which(sim$y == 0)
  idx1 <- which(sim$y == 1)
  
  plot_ly(type = "scatter3d", mode = "markers") %>%
    add_markers(x = Z[idx0, 1], y = Z[idx0, 2], z = Z[idx0, 3],
                name = "Classe 0",
                marker = list(size = marker_size,
                              color = colori[1],
                              opacity = opacity_points)) %>%
    add_markers(x = Z[idx1, 1], y = Z[idx1, 2], z = Z[idx1, 3],
                name = "Classe 1",
                marker = list(size = marker_size,
                              color = colori[2],
                              opacity = opacity_points)) %>%
    layout(title = list(text = titolo, x = 0.05, y = 0.95),
           scene  = list(
             xaxis = list(title = axis[1]),
             yaxis = list(title = axis[2]),
             zaxis = list(title = axis[3])
           ),
           legend = list(orientation = "h", x = 0.35, y = 1.05))
}

shift_fun_corrected <- function(d, livello = c("facile", "medio", "difficile")) {
  livello <- match.arg(livello)
  
  # Nuovi valori base per coordinata
  base_per_coord <- switch(livello,
                           "facile"    = 2.5,   # Aumentato per caso facile
                           "medio"     = 2,   # Valore intermedio
                           "difficile" = 1.8)    # Aumentato per caso difficile
  
  # Esponente modificato: radice decima invece di radice quadrata
  per_coord <- base_per_coord / d^(0.4)  # d^(1/2 - 1/10) = d^0.4
  
  rep(per_coord, d)
}

# 2. Funzione simulazione modificata
simulate_once_corrected <- function(n, d, livello = "medio") {
  shift_vec <- shift_fun_corrected(d, livello)
  y <- sample(0:1, n, replace = TRUE)
  n1 <- sum(y == 0); n2 <- n - n1
  
  mu1 <- rep(0, d)
  mu2 <- shift_vec
  Sig1 <- diag(d)  # Covarianza identità per entrambe
  Sig2 = ar1_cor(d)
  
  X1 <- MASS::mvrnorm(n1, mu1, Sig1)
  X2 <- MASS::mvrnorm(n2, mu2, Sig2)
  
  X <- matrix(NA_real_, n, d)
  X[y == 0, ] <- X1
  X[y == 1, ] <- X2
  list(X = X, y = y)
}

simulate_dataset <- function(type = c("Gauss", "Torus"),
                             n, d,
                             rho     = 0.5,
                             livello = "medio") {
  
  type <- match.arg(type)
  
  if (type == "Gauss") {
    out <- simulate_once_corrected(n, d, livello = livello)
  } else {
    tor <- generate_two_tori(           
      n            = n,
      d            = d,
      livello      = livello,
      noise_factor = 1.5                
    )
    out <- list(X = tor$X, y = tor$y)
  }
  out
}


generate_two_tori <- function(
    n            = 6000L,
    d            = 100L,
    livello      = c("facile", "medio", "difficile"),
    shift_vec    = NULL,
    noise_factor = 1.5,
    seed         = NULL
) {
  ## ────────────────── 0. Controlli preliminari ──────────────────
  stopifnot(d >= 3, n %% 2 == 0)
  if (!is.null(seed)) set.seed(seed)
  
  livello <- match.arg(livello)
  
  ## ────────────────── 1. Vettore di shift ───────────────────────
  if (is.null(shift_vec)) {
    shift_vec <- shift_fun_corrected(d, livello)  # sempre la versione “corrected”
    shift_source <- "corrected"
  } else {
    if (length(shift_vec) != d)
      stop("`shift_vec` deve avere lunghezza d.")
    shift_source <- "user"
  }
  
  norm_shift <- sqrt(sum(shift_vec^2))
  if (norm_shift == 0)
    stop("Il vettore di shift non può essere nullo.")
  
  ## ────────────────── 2. Geometria dei tori ─────────────────────
  R_torus <- norm_shift
  r_torus <- (3 / 7) * norm_shift
  
  n_each <- n / 2L
  angle  <- runif(1, 0, pi / 2)
  
  torus0 <- alphashape3d::rtorus(
    n_each, r = r_torus, R = R_torus,
    ct = c(0, 0, 0), rotx = angle
  )
  torus1 <- alphashape3d::rtorus(
    n_each, r = r_torus, R = R_torus,
    ct = c(R_torus, 0, 0), rotx = angle + pi / 2
  )
  
  X3 <- rbind(torus0, torus1)
  y  <- c(rep(0L, n_each), rep(1L, n_each))
  
  ## ────────────────── 3. Rumore ─────────────────────────────────
  sd_noise <- noise_factor / norm_shift
  if (livello == "facile")
    sd_noise <- sd_noise * 0.5        # metà rumore per il caso più facile
  
  X3 <- X3 + matrix(rnorm(length(X3), 0, sd_noise), ncol = 3)
  
  ## ────────────────── 4. Immersione in ℝᵈ ────────────────────────
  if (d > 3) {
    Q    <- pracma::randortho(d)[, 1:3]  # matrice ortogonale casuale
    Xd   <- X3 %*% t(Q)
    proj <- t(Q)
  } else {
    Xd   <- X3
    proj <- diag(3L)
  }
  
  ## ────────────────── 5. Output ─────────────────────────────────
  list(
    X          = Xd,
    y          = y,
    shift_vec  = shift_vec,
    norm_shift = norm_shift,
    r_used     = r_torus,
    R_used     = R_torus,
    sd_noise   = sd_noise,
    projector  = proj,
    angle      = angle,
    shift_source = shift_source      # “corrected” oppure “user”
  )
}


plot_two_tori <- function(X, y, dims = c(1,2,3),
                          colors = c("#1f77b4", "#ff7f0e"), sz = 2) {
  
  stopifnot(length(dims) == 3, max(dims) <= ncol(X))
  
  idx0 <- which(y == 0); idx1 <- which(y == 1)
  plotly::plot_ly(type = "scatter3d", mode = "markers") %>%
    plotly::add_markers(x = X[idx0, dims[1]], y = X[idx0, dims[2]],
                        z = X[idx0, dims[3]], name = "0",
                        marker = list(size = sz, color = colors[1])) %>%
    plotly::add_markers(x = X[idx1, dims[1]], y = X[idx1, dims[2]],
                        z = X[idx1, dims[3]], name = "1",
                        marker = list(size = sz, color = colors[2])) %>%
    plotly::layout(scene = list(xaxis = list(title = paste0("Dim ", dims[1])),
                                yaxis = list(title = paste0("Dim ", dims[2])),
                                zaxis = list(title = paste0("Dim ", dims[3]))),
                   legend = list(orientation = "h", x = 0.35, y = 1.05))
}

calc_accuracy <- function(pred, truth) {
  mean(pred == truth)
}

eval_reduction <- function(X, y,
                           method    = c("tsne", "umap", "pca"),
                           seed      = NULL,
                           return_Z  = FALSE) {
  
  method <- match.arg(method)
  
  ## -- Embedding
  if (method == "tsne") {
    
    manifold <- reticulate::import("sklearn.manifold", delay_load = TRUE)
    Z <- manifold$TSNE(
      n_components = 2L,
      perplexity   = min(30L, floor((nrow(X) - 1)/3)),
      init         = "pca",
      n_jobs       = -1L
    )$fit_transform(X)
    
  } else if (method == "umap") {
    
    umap_py <- reticulate::import("umap", delay_load = TRUE)
    Z <- umap_py$UMAP(
      n_components = 2L,
      n_neighbors  = 15L,
      min_dist     = 0.1,
      n_jobs       = -1L
    )$fit_transform(X)
    
  } else {                           # PCA
    Z <- run_pca_2d(X)
  }
  
  ## -- Valutazioni (metriche)
  ## 3.1 Albero  → accuracy
  dat <- data.frame(Z1 = Z[,1], Z2 = Z[,2], y = factor(y))
  fit <- rpart::rpart(y ~ ., data = dat, method = "class",
                      control = rpart::rpart.control(cp = 0))
  pred_tree <- as.integer(as.character(predict(fit, dat, type = "class")))
  acc_tree  <- calc_accuracy(pred_tree, y)
  
  ## 3.2 k-means → adjusted Rand Index
  km      <- kmeans(Z, 2, nstart = 10)
  cl      <- ifelse(km$cluster == 2, 1, 0)
  if (mean(cl == y) < 0.5) cl <- 1 - cl           
  ari_km  <- mclust::adjustedRandIndex(cl, y)
  
  #output
  out <- list(
    kmeans = ari_km,          
    tree   = acc_tree         
  )
  if (return_Z) out$Z <- Z
  out
}

make_seed_list <- function(base_seed, n_streams) {
  set.seed(base_seed, kind = "L'Ecuyer-CMRG")
  seeds <- vector("list", n_streams)
  seeds[[1]] <- .Random.seed
  for (i in 2:n_streams)
    seeds[[i]] <- parallel::nextRNGStream(seeds[[i-1]])
  seeds
}

# VISUALIZZAZIONE E SALVATAGGIO DATI GENERATI -----------------------------
# funzione helper: scatter3d plotly pulito, con camera e assi configurabili
make_plotly_scatter3d <- function(X, y,
                                  show_axes = TRUE,
                                  point_size = 2,
                                  opacity_points = 0.8,
                                  colors = c("#1f77b4", "#ff7f0e")) {
  stopifnot(ncol(X) >= 3)
  idx0 <- which(y == 0L); idx1 <- which(y == 1L)
  
  ax <- function(title) {
    if (show_axes) {
      list(title = title, visible = TRUE, showgrid = FALSE,
           zeroline = FALSE, ticks = "outside", showbackground = FALSE)
    } else {
      list(visible = FALSE, showgrid = FALSE, zeroline = FALSE,
           ticks = "", showbackground = FALSE)
    }
  }
  
  plotly::plot_ly(type = "scatter3d", mode = "markers") %>%
    plotly::add_markers(x = X[idx0, 1], y = X[idx0, 2], z = X[idx0, 3],
                        marker = list(size = point_size,
                                      color = colors[1],
                                      opacity = opacity_points),
                        hoverinfo = "skip", showlegend = FALSE) %>%
    plotly::add_markers(x = X[idx1, 1], y = X[idx1, 2], z = X[idx1, 3],
                        marker = list(size = point_size,
                                      color = colors[2],
                                      opacity = opacity_points),
                        hoverinfo = "skip", showlegend = FALSE) %>%
    plotly::layout(
      scene = list(
        xaxis = ax("x"), yaxis = ax("y"), zaxis = ax("z"),
        aspectmode = "cube",
        camera = list(eye = list(x = 1.5, y = 1.5, z = 1.2))
      ),
      showlegend = FALSE,
      margin = list(l = 0, r = 0, b = 0, t = 0),
      paper_bgcolor = "rgba(0,0,0,0)",
      plot_bgcolor  = "rgba(0,0,0,0)"
    )
}

# ─────────────────────────────────────────────────────────────
# 4 grafici (20k punti, d = 3). Esporta a mano dal viewer.
# ─────────────────────────────────────────────────────────────

# gaussiani — facile
sim_g_facile <- simulate_once_corrected(n = 10000L, d = 3L, livello = "facile")
p_g_facile   <- make_plotly_scatter3d(sim_g_facile$X, sim_g_facile$y, show_axes = TRUE)
p_g_facile

# gaussiani — difficile
sim_g_diff <- simulate_once_corrected(n = 10000L, d = 3L, livello = "difficile")
p_g_diff   <- make_plotly_scatter3d(sim_g_diff$X, sim_g_diff$y, show_axes = TRUE)
p_g_diff

# due tori — facile
tor_facile <- generate_two_tori(n = 10000L, d = 3L, livello = "facile")
p_t_facile <- make_plotly_scatter3d(tor_facile$X, tor_facile$y, show_axes = TRUE)
p_t_facile

# due tori — difficile
tor_diff <- generate_two_tori(n = 10000L, d = 3L, livello = "difficile")
p_t_diff <- make_plotly_scatter3d(tor_diff$X, tor_diff$y, show_axes = TRUE)
p_t_diff



# CICLO PRINCIPALE DI SIMULAZIONE, parallelizzato -------------------------
run_simulation <- function(
    ns      = c(100, 1000, 10000),
    ds      = c(10, 50, 100),
    livelli = c("facile", "medio", "difficile"),
    reps    = 1,
    rho     = 0.5,
    res_dt  = data.table::data.table(),
    skip_completed = TRUE,
    append_file    = NULL,
    data_types     = c("Gauss", "Torus")
) {
  
  ## ───────────────────── 0. Pre-requisiti ──────────────────────
  res_dt <- data.table::as.data.table(res_dt)
  if ("shift_version" %in% names(res_dt))
    res_dt[, shift_version := NULL]          # pulizia di dataset “legacy”
  
  overwrite_csv <- !is.null(append_file) && nrow(res_dt) == 0L
  if (overwrite_csv && file.exists(append_file)) file.remove(append_file)
  first_write <- overwrite_csv
  
  skipped_sets <- data.table::data.table(
    n = integer(), d = integer(), livello = character()
  )
  
  .already_done <- function(n_val, d_val, lvl_val) {
    if (!skip_completed || nrow(res_dt) == 0L) return(FALSE)
    res_dt[n == n_val & d == d_val & livello == lvl_val, .N] > 0L
  }
  
  total_sets <- length(ns) * length(ds) * length(livelli)
  set_i      <- 0L
  
  ## ───────────────────── 1. Ciclo sui parametri ─────────────────────
  for (n in ns) for (d in ds) for (lvl in livelli) {
    
    set_i <- set_i + 1L
    
    if (.already_done(n, d, lvl)) {
      message(sprintf(
        "--- Set %d/%d (n=%d, d=%d, livello=%s) già presente – salto.",
        set_i, total_sets, n, d, lvl
      ))
      skipped_sets <- rbind(skipped_sets,
                            data.table(n = n, d = d, livello = lvl))
      next
    }
    
    lbl <- sprintf(
      "Set %d/%d – n=%d, d=%d, livello=%s, reps=%d",
      set_i, total_sets, n, d, lvl, reps
    )
    tictoc::tic(lbl)
    
    ## ───── 2. Parallel loop sulle ripetizioni ─────
    seeds <- make_seed_list(base_seed = n + d, n_streams = reps)
    res_list <- future.apply::future_lapply(
      X = seq_len(reps),
      FUN = function(r) {
        
        require(data.table); require(MASS); require(rpart)
        require(reticulate);  require(mclust)
        
        out_all <- vector("list", length(data_types))
        
        for (t_ix in seq_along(data_types)) {
          
          dtype <- data_types[t_ix]
          
          ## 2.1 Generazione dati
          sim <- simulate_dataset(
            type    = dtype,
            n       = n,
            d       = d,
            rho     = rho,
            livello = lvl
          )
          
          Xnum <- as.matrix(sim$X)
          storage.mode(Xnum) <- "double"
          
          ## 2.2 Valutazione riduzioni
          acc_t <- eval_reduction(Xnum, sim$y, "tsne")
          acc_u <- eval_reduction(Xnum, sim$y, "umap")
          acc_p <- eval_reduction(Xnum, sim$y, "pca")
          
          ## 2.3 Assemblaggio risultati
          out_all[[t_ix]] <- data.table::rbindlist(list(
            # t-SNE
            data.table(data_type = dtype, n = n, d = d, livello = lvl,
                       rep = r, method = "tsne", measure = "kmeans",
                       `acc./ARI` = round(acc_t$kmeans, 4)),
            data.table(data_type = dtype, n = n, d = d, livello = lvl,
                       rep = r, method = "tsne", measure = "tree",
                       `acc./ARI` = round(acc_t$tree, 4)),
            # UMAP
            data.table(data_type = dtype, n = n, d = d, livello = lvl,
                       rep = r, method = "umap", measure = "kmeans",
                       `acc./ARI` = round(acc_u$kmeans, 4)),
            data.table(data_type = dtype, n = n, d = d, livello = lvl,
                       rep = r, method = "umap", measure = "tree",
                       `acc./ARI` = round(acc_u$tree, 4)),
            # PCA
            data.table(data_type = dtype, n = n, d = d, livello = lvl,
                       rep = r, method = "pca", measure = "kmeans",
                       `acc./ARI` = round(acc_p$kmeans, 4)),
            data.table(data_type = dtype, n = n, d = d, livello = lvl,
                       rep = r, method = "pca", measure = "tree",
                       `acc./ARI` = round(acc_p$tree, 4))
          ))
        }
        
        data.table::rbindlist(out_all)
      },
      future.seed     = seeds,  
      future.globals  = TRUE,
      future.packages = c("data.table", "MASS", "rpart",
                          "reticulate", "mclust")
    )
    
    ## ───── 3. Accumulo risultati & CSV ─────
    block_dt <- data.table::rbindlist(res_list)
    res_dt   <- rbind(res_dt, block_dt)
    
    if (!is.null(append_file)) {
      data.table::fwrite(block_dt, file = append_file, append = !first_write)
      first_write <- FALSE
    }
    
    sec <- { tm <- tictoc::toc(quiet = TRUE); tm$toc - tm$tic }
    
    message(sprintf(
      ">>> Completato: reps=%d, n=%d, d=%d, livello=%s | %.2f s",
      reps, n, d, lvl, sec
    ))
    if (n == 10000L) {
      try(ntfy_send(sprintf(
        "Run completata – reps=%d, n=%d, d=%d, livello=%s, t=%.2f s",
        reps, n, d, lvl, sec)), silent = TRUE)
    }
  }
  
  ## ───── 4. Log finale ─────
  if (nrow(skipped_sets)) {
    message("\nParametri già presenti (saltati):")
    print(unique(skipped_sets[order(n, d, livello)]))
  } else {
    message("\nNessun set di parametri era già presente: eseguito tutto da zero.")
  }
  
  ## ───── 5. Ordinamento colonne ─────
  if (nrow(res_dt)) {
    data.table::setcolorder(
      res_dt,
      c("data_type", "n", "d", "livello", "rep",
        "method", "measure", "acc./ARI")
    )
  }
  
  invisible(res_dt)
}

# CICLO PRINCIPALE SIMULAZIONE, NON parallelizzato ------------------------
run_simulation_np <- function(
    ns      = c(100, 1000, 10000),
    ds      = c(10, 50, 100),
    livelli = c("facile", "medio", "difficile"),
    reps    = 1,
    rho     = 0.5,
    res_dt  = data.table::data.table(),
    skip_completed = TRUE,
    append_file    = NULL,
    data_types     = c("Gauss", "Torus")
) {
  
  ## ───────────────────── 0. Pre-requisiti ─────────────────────
  res_dt <- data.table::as.data.table(res_dt)
  if ("shift_version" %in% names(res_dt))
    res_dt[, shift_version := NULL]          # pulizia dataset “legacy”
  
  overwrite_csv <- !is.null(append_file) && nrow(res_dt) == 0L
  if (overwrite_csv && file.exists(append_file)) file.remove(append_file)
  first_write <- overwrite_csv
  
  skipped_sets <- data.table::data.table(
    n = integer(), d = integer(), livello = character()
  )
  
  .already_done <- function(n_val, d_val, lvl_val) {
    if (!skip_completed || nrow(res_dt) == 0L) return(FALSE)
    res_dt[n == n_val & d == d_val & livello == lvl_val, .N] > 0L
  }
  
  total_sets <- length(ns) * length(ds) * length(livelli)
  set_i      <- 0L
  
  ## ───────────────────── 1. Ciclo sui parametri ───────────────
  for (n in ns) for (d in ds) for (lvl in livelli) {
    
    set_i <- set_i + 1L
    
    if (.already_done(n, d, lvl)) {
      message(sprintf(
        "--- Set %d/%d (n=%d, d=%d, livello=%s) già presente – salto.",
        set_i, total_sets, n, d, lvl
      ))
      skipped_sets <- rbind(
        skipped_sets, data.table(n = n, d = d, livello = lvl)
      )
      next
    }
    
    lbl <- sprintf(
      "Set %d/%d – n=%d, d=%d, livello=%s, reps=%d",
      set_i, total_sets, n, d, lvl, reps
    )
    tictoc::tic(lbl)
    
    ## ───── 2. Loop SEQUENZIALE sulle repliche ─────
    res_list <- vector("list", reps)
    
    for (r in seq_len(reps)) {
      
      ## 2.0 Seme per la replica r
      set.seed(n + d + r)
      
      out_all <- vector("list", length(data_types))
      
      for (t_ix in seq_along(data_types)) {
        
        dtype <- data_types[t_ix]
        
        ## 2.1 Generazione dati
        sim <- simulate_dataset(
          type    = dtype,
          n       = n,
          d       = d,
          rho     = rho,
          livello = lvl
        )
        
        Xnum <- as.matrix(sim$X)
        storage.mode(Xnum) <- "double"
        
        ## 2.2 Valutazione riduzioni
        acc_t <- eval_reduction(Xnum, sim$y, "tsne")
        acc_u <- eval_reduction(Xnum, sim$y, "umap")
        acc_p <- eval_reduction(Xnum, sim$y, "pca")
        
        ## 2.3 Assemblaggio risultati
        out_all[[t_ix]] <- data.table::rbindlist(list(
          # t-SNE
          data.table(data_type = dtype, n = n, d = d, livello = lvl,
                     rep = r, method = "tsne", measure = "kmeans",
                     `acc./ARI` = round(acc_t$kmeans, 4)),
          data.table(data_type = dtype, n = n, d = d, livello = lvl,
                     rep = r, method = "tsne", measure = "tree",
                     `acc./ARI` = round(acc_t$tree, 4)),
          # UMAP
          data.table(data_type = dtype, n = n, d = d, livello = lvl,
                     rep = r, method = "umap", measure = "kmeans",
                     `acc./ARI` = round(acc_u$kmeans, 4)),
          data.table(data_type = dtype, n = n, d = d, livello = lvl,
                     rep = r, method = "umap", measure = "tree",
                     `acc./ARI` = round(acc_u$tree, 4)),
          # PCA
          data.table(data_type = dtype, n = n, d = d, livello = lvl,
                     rep = r, method = "pca", measure = "kmeans",
                     `acc./ARI` = round(acc_p$kmeans, 4)),
          data.table(data_type = dtype, n = n, d = d, livello = lvl,
                     rep = r, method = "pca", measure = "tree",
                     `acc./ARI` = round(acc_p$tree, 4))
        ))
      } # fine ciclo data_types
      
      res_list[[r]] <- data.table::rbindlist(out_all)
    } # fine ciclo reps
    
    ## ───── 3. Accumulo risultati & CSV ─────
    block_dt <- data.table::rbindlist(res_list)
    res_dt   <- rbind(res_dt, block_dt)
    
    if (!is.null(append_file)) {
      data.table::fwrite(block_dt, file = append_file, append = !first_write)
      first_write <- FALSE
    }
    
    sec <- { tm <- tictoc::toc(quiet = TRUE); tm$toc - tm$tic }
    
    message(sprintf(
      ">>> Completato: reps=%d, n=%d, d=%d, livello=%s | %.2f s, %.2f min",
      reps, n, d, lvl, sec, sec/60
    ))
    if (n == 10000L) {
      try(ntfy_send(sprintf(
        "Run completata – reps=%d, n=%d, d=%d, livello=%s, t=%.2f s, min=%.2f",
        reps, n, d, lvl, sec, sec/60)), silent = TRUE)
    }
  } # fine loop parametri
  
  ## ───── 4. Log finale ─────
  if (nrow(skipped_sets)) {
    message("\nParametri già presenti (saltati):")
    print(unique(skipped_sets[order(n, d, livello)]))
  } else {
    message("\nNessun set di parametri era già presente: eseguito tutto da zero.")
  }
  
  ## ───── 5. Ordinamento colonne ─────
  if (nrow(res_dt)) {
    data.table::setcolorder(
      res_dt,
      c("data_type", "n", "d", "livello", "rep",
        "method", "measure", "acc./ARI")
    )
  }
  
  invisible(res_dt)
}

# LANCIO SIMULAZIONE ------------------------------------------------------
library(data.table)
library(future.apply)
library(tictoc)

tictoc::tic("Simulazione – versione non parallela")
ris <- run_simulation_np(
  ns       = c(100, 1000, 10000),#
  ds       = c(10, 50, 100),#
  livelli  = c("facile", "difficile"),
  reps     = 100,
  data_types = c("Gauss", "Torus"),
  append_file = file.path(getwd(), "risultati_sim_np.csv")
)
tictoc::toc()
beepr::beep(8)
ntfy_send("Fine della run unica non parallela.")


# ANALISI DEI RISULTATI NUOVO ---------------------------------------------
## ─────────────────────  Librerie  ─────────────────────
library(data.table)
library(tidyverse)
library(ggh4x)
library(scales)

## ───────────────────── 1. Lettura & fattori ───────────
ris <- fread("risultati_sim.csv") %>%
  mutate(
    # 1.1 dataset / riduzione / misura
    data_type = factor(data_type,
                       levels = c("Gauss", "Torus"),
                       labels = c("Gaussiani", "Tori")),
    measure   = factor(measure,
                       levels = c("tree", "kmeans"),
                       labels = c("Albero", "k-means")),
    method    = factor(method,
                       levels = c("pca", "umap", "tsne"),
                       labels = c("PCA", "UMAP", "t-SNE")),
    # 1.2 dimensione campionaria n
    n = factor(n,
               levels = sort(unique(n)),
               labels = paste0("n = ", comma(sort(unique(n)), big.mark = " "))),
    # 1.3 dimensione d
    d = factor(d,
               levels = sort(unique(d)),
               labels = paste0("d = ", sort(unique(d)))),
    # 1.4 livello di difficoltà
    livello = factor(livello,
                     levels = c("facile", "medio", "difficile"),
                     labels = c("Facile", "Medio", "Difficile"))
  )

glimpse(ris)

## ───────────────────── 2. Funzioni di supporto ────────
# (opzionale) filtra 1-2 livelli di difficoltà
prep_livello_data <- function(df, misura,
                              livelli_keep = c("Facile", "Difficile")) {
  df %>%
    filter(measure == misura, livello %in% livelli_keep) %>%
    mutate(livello = fct_drop(livello))
}

# wrapper generico per box-plot con facet annidato
# modifiche: niente titolo; strip di riga (d) spostate a sinistra e all'esterno
plot_nested <- function(df, x_var, row_vars, col_vars) {
  ggplot(df, aes(x = {{ x_var }}, y = `acc./ARI`, fill = method)) +
    geom_blank() +
    geom_boxplot(width = .7, outlier.size = .6) +
    facet_nested(
      rows   = {{ row_vars }},
      cols   = {{ col_vars }},
      scales = "free_x",
      space  = "free_x",
      switch = "y"                 # strip di riga (d) a sinistra
    ) +
    scale_fill_brewer(palette = "Set2", guide = "none") +
    labs(x = NULL, y = NULL) +
    theme_bw(base_size = 16) +     # tutto a 16 pt, non bold
    theme(
      strip.placement   = "outside",
      strip.text.x      = element_text(size = 16, face = "plain"),
      strip.text.y      = element_text(size = 16, face = "plain"),
      strip.text.y.left = element_text(size = 16, face = "plain",
                                       angle = 90,                 # "d = 10, ..." in verticale
                                       margin = margin(r = 6)),
      axis.text.x       = element_text(angle = 45, hjust = 1)      # non forzo la size dei tick
      # axis.text.y: lasciato di default (prende il base_size)
    )
}

## ───────────────────── 3. Grafici “classici” g1–g4 ────
plot_dimred <- function(df) {
  plot_nested(
    df,
    x_var    = method,
    row_vars = vars(d),
    col_vars = vars(n, livello)    # livello al posto di shift
  )
}

g1 <- plot_dimred(ris %>% filter(data_type == "Gaussiani",
                                 measure    == "Albero"))
g2 <- plot_dimred(ris %>% filter(data_type == "Gaussiani",
                                 measure    == "k-means"))
g3 <- plot_dimred(ris %>% filter(data_type == "Tori",
                                 measure    == "Albero"))
g4 <- plot_dimred(ris %>% filter(data_type == "Tori",
                                 measure    == "k-means"))

print(g1); print(g2); print(g3); print(g4)

## ───────────────────── 4. Salvataggio PDF 18×12 cm ───
out_dir <- "output"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

save_pdf <- function(plot_obj, filename, w = 18, h = 12) {
  ggsave(
    filename = file.path(out_dir, filename),
    plot     = plot_obj,
    device   = cairo_pdf
  )
}

save_pdf(g1, "ris_sim_gaussiani_albero.pdf")
save_pdf(g2, "ris_sim_gaussiani_kmeans.pdf")
save_pdf(g3, "ris_sim_tori_albero.pdf")
save_pdf(g4, "ris_sim_tori_kmeans.pdf")

# VISUALIZZAZIONI -----------------------------------------------------
rm(list = ls(all.names = TRUE), envir = .GlobalEnv)
gc()

pkgs <- c("data.table","RSpectra","reticulate","rpart","mclust",
          "ggplot2","FNN","pracma","alphashape3d","ggh4x")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))

PY311 <- "C:/Users/simop/AppData/Local/Programs/Python/Python311/python.exe"
Sys.setenv(RETICULATE_PYTHON = PY311)
reticulate::use_python(PY311, required = TRUE)

umap_py  <- reticulate::import("umap")
manifold <- reticulate::import("sklearn.manifold")

# PARAMETRI SIMULAZIONE 
n       <- 5000
ds      <- c(10L, 50L, 100L)
livelli <- c("facile", "difficile")
types   <- c("Gauss", "Torus")              # entrambe le famiglie
outfile <- "embedding_corr.csv"             # unica versione richiesta

# 1) FUNZIONI

# matrice di correlazione AR(1)
ar1_cor <- function(d, rho = 0.5) {
  expo <- abs(outer(0:(d-1), 0:(d-1), "-"))
  rho^expo
}

# shift "corrected" (vettore delle traslazioni coordinata per coordinata)
shift_fun_corrected <- function(d, livello = c("facile","medio","difficile")){
  livello <- match.arg(livello)
  base <- switch(livello,
                 "facile"    = 2.5,
                 "medio"     = 2.0,
                 "difficile" = 1.8)
  rep(base / d^0.4, d)
}

# gaussiani (soluzione corrected)
simulate_once_corrected <- function(n, d, livello = "medio"){
  shift_vec <- shift_fun_corrected(d, livello)
  y  <- sample(0:1, n, replace = TRUE)
  n1 <- sum(y == 0); n2 <- n - n1
  mu1 <- rep(0, d);  mu2 <- shift_vec
  Sig1 <- diag(d);   Sig2 <- ar1_cor(d, 0.5)
  X1 <- MASS::mvrnorm(n1, mu1, Sig1)
  X2 <- MASS::mvrnorm(n2, mu2, Sig2)
  X <- matrix(NA_real_, n, d); X[y==0,] <- X1; X[y==1,] <- X2
  list(X = X, y = y)
}

# due tori immersi in R^d (geometria regolata dallo shift corrected)
generate_two_tori <- function(n, d, livello = "medio", noise_factor = 1.5){
  shift_vec  <- shift_fun_corrected(d, livello)
  norm_shift <- sqrt(sum(shift_vec^2))
  stopifnot(d >= 3, n %% 2 == 0, norm_shift > 0)
  
  R <- norm_shift
  r <- (3/7) * norm_shift
  n_each <- n / 2L
  angle  <- runif(1, 0, pi/2)
  
  torus0 <- alphashape3d::rtorus(n_each, r, R, ct = c(0,0,0),   rotx = angle)
  torus1 <- alphashape3d::rtorus(n_each, r, R, ct = c(R,0,0),   rotx = angle + pi/2)
  X3 <- rbind(torus0, torus1)
  y  <- c(rep(0L, n_each), rep(1L, n_each))
  
  sd_noise <- noise_factor / norm_shift
  if (livello == "facile") sd_noise <- sd_noise * 0.5
  X3 <- X3 + matrix(rnorm(length(X3), 0, sd_noise), ncol = 3)
  
  if (d > 3){
    Q  <- pracma::randortho(d)[,1:3]
    Xd <- X3 %*% t(Q)
  } else Xd <- X3
  list(X = Xd, y = y)
}

# PCA 2-D (centro e SVD parziale)
run_pca2d <- function(X){
  Xc <- scale(X, center = TRUE, scale = FALSE)
  sv <- RSpectra::svds(Xc, k = 2)
  sv$u %*% diag(sv$d)
}

# calcolo embedding + metriche (accuracy albero, ARI k-means)
eval_emb <- function(X, y, method = c("pca","tsne","umap")){
  method <- match.arg(method)
  Z <- switch(method,
              pca  = run_pca2d(X),
              tsne = manifold$TSNE(n_components = 2L,
                                   perplexity   = min(30L, floor((nrow(X)-1)/3)),
                                   init = "pca")$fit_transform(X),
              umap = umap_py$UMAP(n_components = 2L,
                                  n_neighbors  = 15L,
                                  min_dist     = 0.1)$fit_transform(X))
  
  dat <- data.frame(Z1 = Z[,1], Z2 = Z[,2], y = factor(y))
  acc <- mean(predict(rpart(y ~ ., data = dat, method = "class",
                            control = rpart.control(cp = 0)),
                      dat, type = "class") == dat$y)
  km  <- kmeans(Z, 2, nstart = 10)$cluster
  if (mean(km == y) < .5) km <- 3 - km
  ari <- mclust::adjustedRandIndex(km, y)
  list(Z = Z, accuracy = acc, ari = ari)
}

# 2) SIMULAZIONE “CORRECTED” + CSV
emb_list <- list(); ix <- 1L
for (lvl in livelli){
  for (dtype in types){
    for (d in ds){
      cat(sprintf("→ %s | d=%d | livello=%s …\n", dtype, d, lvl))
      sim <- if (dtype == "Gauss"){
        simulate_once_corrected(n, d, livello = lvl)
      } else {
        generate_two_tori(n, d, livello = lvl)
      }
      for (mtd in c("pca","tsne","umap")){
        res <- eval_emb(sim$X, sim$y, mtd)
        emb_list[[ix]] <- data.table(
          livello   = lvl,
          data_type = dtype,
          d         = d,
          method    = mtd,
          dim1      = res$Z[,1],
          dim2      = res$Z[,2],
          y         = sim$y,
          accuracy  = round(res$accuracy, 4),
          ari       = round(res$ari, 4)
        )
        ix <- ix + 1L
        cat(mtd, "\n")
      }
    }
  }
}
emb_dt <- data.table::rbindlist(emb_list, use.names = TRUE)
if (file.exists(outfile)) file.remove(outfile)
data.table::fwrite(emb_dt, file = outfile)
cat("File scritto in:", normalizePath(outfile), "\n")
ntfy_send("Fine run ciclo")

# 3) GRAFICI MINIMALI + SALVATAGGIO

# funzione grafico minimal con strip di riga a sinistra
plot_grid_minimal <- function(dt){
  if (nrow(dt) == 0L) return(NULL)
  
  ggplot(dt, aes(dim1, dim2, colour = factor(y))) +
    geom_point(alpha = .6, size = .6) +
    ggh4x::facet_grid2(
      rows  = vars(d),
      cols  = vars(method),
      scales = "free",
      independent = "all",
      switch = "y"               # etichette delle righe a sinistra
    ) +
    scale_colour_manual(values = c("#1f77b4","#ff7f0e"), name = "Classe") +
    theme_bw(base_size = 11) +
    theme(
      legend.position = "none",        
      axis.title  = element_blank(),
      axis.text   = element_blank(),
      axis.ticks  = element_blank(),
      panel.grid  = element_blank(),
      strip.placement = "outside",
      strip.text.x = element_text(size = 20, face = "plain"),
      strip.text.y.left = element_text(size = 20, face = "plain"),
      plot.margin = margin(6, 6, 6, 6)
    )
}

# helper per salvare quadrato 9x9 in pdf nella cartella output
save_square_pdf <- function(p, filename_base){
  if (!dir.exists("output")) dir.create("output")
  ggsave(
    filename = file.path("output", paste0(filename_base, ".pdf")),
    plot = p, device = cairo_pdf, width = 9, height = 9, units = "in"
  )
}

# lettura e normalizzazione fattori
emb <- data.table::fread(outfile)
emb[, method := factor(method,
                       levels = c("pca","tsne","umap"),
                       labels = c("PCA","t-SNE","UMAP"))]
emb[, d := factor(d, levels = sort(unique(d)),
                  labels = paste0("d = ", sort(unique(d))))]

# 4 grafici
g_gauss_f <- plot_grid_minimal(emb[data_type=="Gauss" & livello=="facile"])
g_torus_f <- plot_grid_minimal(emb[data_type=="Torus" & livello=="facile"])
g_gauss_d <- plot_grid_minimal(emb[data_type=="Gauss" & livello=="difficile"])
g_torus_d <- plot_grid_minimal(emb[data_type=="Torus" & livello=="difficile"])

# salvataggio pdf 9×9 nella cartella "output"
if (!is.null(g_gauss_f)) save_square_pdf(g_gauss_f, "emb_gauss_facile")
if (!is.null(g_torus_f)) save_square_pdf(g_torus_f, "emb_tori_facile")
if (!is.null(g_gauss_d)) save_square_pdf(g_gauss_d, "emb_gauss_difficile")
if (!is.null(g_torus_d)) save_square_pdf(g_torus_d, "emb_tori_difficile")

# ANALISI RISULTATI vecchio --------------------------------------------------------

# ---- Pacchetti
library(tidyverse)      
library(knitr)          
library(kableExtra)     # tabelle in LaTeX / HTML
library(viridis)        # palette per heatmap

# ---- 1. Lettura e pre‑processing
ris <- read_csv(
  "risultati_sim.csv",
  show_col_types = FALSE,
  col_types = cols(
    data_type = col_character(),       
    n         = col_integer(),
    d         = col_integer(),
    shift     = col_double(),
    rep       = col_integer(),
    method    = col_character(),
    measure   = col_character(),
    accuracy  = col_double(),
    recall    = col_double(),
    precision = col_double(),
    f1        = col_double()
  )
) %>% 
  mutate(
    data_type = factor(data_type,
                       levels = c("Gauss", "Taurus"),
                       labels = c("Gaussiani", "Tori")),
    method    = factor(method,
                       levels = c("pca", "tsne", "umap"),
                       labels = c("PCA", "t‑SNE", "UMAP")),
    measure   = factor(measure,
                       levels = c("kmeans", "tree"),
                       labels = c("k‑means", "Albero"))
  )

glimpse(ris)

# ---- 2. Sommario globale
# (una riga per data_type × metodo × misura)
global_summary <- ris %>% 
  group_by(data_type, method, measure) %>% 
  summarise(across(accuracy:f1,
                   list(media = mean, sd = sd),
                   .names = "{.col}_{.fn}"),
            n_osservazioni = n(),
            .groups = "drop")

# Tabella formattata (LaTeX)
global_summary %>% 
  mutate(across(where(is.numeric), round, 3)) %>% 
  kable(
    format   = "latex",
    booktabs = TRUE,
    caption  = "Sommario globale delle metriche per metodo, misura e tipo di dato",
    col.names = c("Dataset", "Metodo", "Misura",
                  "Acc. media", "Acc. sd",
                  "Rec. media", "Rec. sd",
                  "Prec. media", "Prec. sd",
                  "F1 media",  "F1 sd",
                  "N")
  ) %>% 
  kable_styling(latex_options = c("striped", "hold_position", "scale_down"),
                full_width    = FALSE,
                position      = "center") %>% 
  collapse_rows(columns = 1:3, valign = "top")

# ---- 3. Distribuzione globale delle metriche
ris_long <- ris %>% 
  pivot_longer(cols = accuracy:f1,
               names_to  = "metrica",
               values_to = "valore")

# Filtra ordine dei pannelli (row = metrica, col = measure) con facet per data_type
ris_long <- ris_long %>% mutate(metrica = factor(metrica, levels = c("accuracy", "recall", "precision", "f1")))

ggplot(ris_long, aes(method, valore, fill = method)) +
  geom_violin(trim = FALSE, alpha = .45) +
  geom_boxplot(width = .1, outlier.shape = NA) +
  facet_grid(data_type ~ metrica + measure, scales = "free_y") +
  labs(title = "Distribuzione delle metriche per metodo di riduzione e dataset",
       x = "Metodo", y = "Valore") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none",
        strip.text.y = element_text(angle = 0))

# ---- 4. Analisi stratificata: parametro n
by_n <- ris %>% 
  group_by(data_type, n, method, measure) %>% 
  summarise(across(accuracy:f1, mean), .groups = "drop")

# Tabella
by_n %>% 
  mutate(across(accuracy:f1, ~ round(.x, 3))) %>% 
  arrange(data_type, measure, method, n) %>% 
  kable(
    format   = "latex",
    booktabs = TRUE,
    caption  = "Metriche medie stratificate per n (per dataset)",
    col.names = c("Dataset", "n", "Metodo", "Misura", "Accuracy", "Recall", "Precision", "F1")
  ) %>% 
  kable_styling(latex_options = c("hold_position", "scale_down"),
                full_width    = FALSE) %>% 
  collapse_rows(columns = 1:4, valign = "top")

# Grafico
ggplot(by_n, aes(factor(n), accuracy, colour = method, group = method)) +
  geom_line(linewidth = .9) +
  geom_point(size = 3) +
  facet_grid(data_type ~ measure) +
  labs(title = "Accuracy media al variare di n",
       x = "n (osservazioni)", y = "Accuracy media") +
  theme_minimal(base_size = 13)

# ---- 5. Analisi stratificata: parametro d
by_d <- ris %>% 
  group_by(data_type, d, method, measure) %>% 
  summarise(across(accuracy:f1, mean), .groups = "drop")

# Tabella
by_d %>% 
  mutate(across(accuracy:f1, ~ round(.x, 3))) %>% 
  arrange(data_type, measure, method, d) %>% 
  kable(
    format   = "latex",
    booktabs = TRUE,
    caption  = "Metriche medie stratificate per d (per dataset)",
    col.names = c("Dataset", "d", "Metodo", "Misura", "Accuracy", "Recall", "Precision", "F1")
  ) %>% 
  kable_styling(latex_options = c("hold_position", "scale_down"),
                full_width    = FALSE) %>% 
  collapse_rows(columns = 1:4, valign = "top")

# Grafico
ggplot(by_d, aes(factor(d), accuracy, colour = method, group = method)) +
  geom_line(linewidth = .9) +
  geom_point(size = 3) +
  facet_grid(data_type ~ measure) +
  labs(title = "Accuracy media al variare di d",
       x = "d (dimensioni)", y = "Accuracy media") +
  theme_minimal(base_size = 13)

# ---- 6. Analisi stratificata: parametro shift
by_shift <- ris %>% 
  group_by(data_type, shift, method, measure) %>% 
  summarise(across(accuracy:f1, mean), .groups = "drop")

# Tabella
by_shift %>% 
  mutate(across(accuracy:f1, ~ round(.x, 3))) %>% 
  arrange(data_type, measure, method, shift) %>% 
  kable(
    format   = "latex",
    booktabs = TRUE,
    caption  = "Metriche medie stratificate per shift (per dataset)",
    col.names = c("Dataset", "Shift", "Metodo", "Misura", "Accuracy", "Recall", "Precision", "F1")
  ) %>% 
  kable_styling(latex_options = c("hold_position", "scale_down"),
                full_width    = FALSE) %>% 
  collapse_rows(columns = 1:4, valign = "top")

# Grafico
# Nota: nei Gaussiani ci si attende una tendenza crescente, nei Tori potenzialmente decrescente.
ggplot(by_shift, aes(shift, accuracy, colour = method, group = method)) +
  geom_line(linewidth = .9) +
  geom_point(size = 3) +
  facet_grid(data_type ~ measure) +
  labs(title = "Accuracy media al variare di shift",
       subtitle = "Il significato di shift dipende dal dataset: distanze fra medie (Gaussiani) vs spessore dei tori (Tori)",
       x = "Shift", y = "Accuracy media") +
  theme_minimal(base_size = 13)

# ---- 7. Heatmap n × d dell’accuracy
heat_nd <- ris %>% 
  group_by(data_type, n, d, method, measure) %>% 
  summarise(accuracy = mean(accuracy), .groups = "drop")

heat_nd %>% 
  mutate(across(c(n, d), factor),
         accuracy = round(accuracy, 3)) %>% 
  arrange(data_type, method, measure, n, d) %>% 
  kable(
    format   = "latex",
    booktabs = TRUE,
    caption  = "Accuracy media per combinazione (n, d) – stratificata per dataset, metodo e misura",
    col.names = c("Dataset", "n", "d", "Metodo", "Misura", "Accuracy")
  ) %>% 
  kable_styling(latex_options = c("striped", "hold_position", "scale_down"),
                full_width    = FALSE,
                position      = "center") %>% 
  collapse_rows(columns = 1:4, valign = "top")

# Heatmap grafica
ggplot(heat_nd, aes(factor(d), factor(n), fill = accuracy)) +
  geom_tile(colour = "white") +
  facet_grid(data_type + method ~ measure) +
  scale_fill_viridis_c(option = "D") +
  labs(title = "Heatmap dell’accuracy media (n vs d)",
       subtitle = "Facette per dataset e metodo",   
       x = "d (dimensioni)", y = "n (osservazioni)") +
  theme_minimal(base_size = 13)

# ---- 8. Metodo vincente per configurazione
winners <- ris %>% 
  group_by(data_type, n, d, shift, rep, measure) %>% 
  slice_max(order_by = accuracy, n = 1, with_ties = FALSE) %>% 
  ungroup()

winner_counts <- winners %>% 
  count(data_type, method) %>% 
  group_by(data_type) %>% 
  mutate(percentuale = 100 * n / sum(n)) %>% 
  ungroup()

# Tabella
winner_counts %>% 
  mutate(percentuale = round(percentuale, 1)) %>% 
  arrange(data_type, desc(percentuale)) %>% 
  kable(
    format   = "latex",
    booktabs = TRUE,
    caption  = "Percentuale di configurazioni (per dataset) in cui ciascun metodo ottiene l’accuracy massima",
    col.names = c("Dataset", "Metodo", "Vittorie", "%")
  ) %>% 
  kable_styling(latex_options = c("hold_position"),
                full_width    = FALSE) %>% 
  collapse_rows(columns = 1, valign = "top")

# Grafico
ggplot(winner_counts, aes(reorder(method, -percentuale), percentuale, fill = method)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.1f%%", percentuale)), vjust = -0.2, size = 4.5) +
  facet_wrap(~ data_type, nrow = 1) +
  labs(title = "Frequenza di \u00ABvittoria\u00BB (accuracy più alta) per metodo e dataset",
       x = "Metodo", y = "Percentuale di vittorie") +
  ylim(0, 100) +
  theme_minimal(base_size = 13)



# UMAP 3 dimensioni -------------------------------------------------------
set.seed(1234)
sim <- simulate_once(n = 1000, d = 3, rho = 0.4, shift = 2)
X   <- sim$X                               # 300 × 3
y   <- factor(sim$y)

## 1. Embedding UMAP
emb <- umap_py$UMAP(
  n_components = 2L, n_neighbors = 15L,
  min_dist = 0.1, random_state = as.integer(123L)
)$fit_transform(X) |> as.matrix()

## 2. Piano dai minimi quadrati
# matrice modello M = [1  u1  u2]
M  <- cbind(1, emb)                        # 300 × 3
B  <- solve(t(M) %*% M, t(M) %*% X)        # 3 × 3 coefficenti (intercetta + 2 slope)

pred <- M %*% B                            # punti proiettati (300 × 3)

## griglia regolare nei due assi UMAP
u <- seq(min(emb[,1]), max(emb[,1]), length = 30)
v <- seq(min(emb[,2]), max(emb[,2]), length = 30)
UV <- expand.grid(u = u, v = v)
Mg <- cbind(1, UV$u, UV$v)                 # 900 × 3
plane <- Mg %*% B                          # 900 × 3
xg <- matrix(plane[,1], 30, 30)
yg <- matrix(plane[,2], 30, 30)
zg <- matrix(plane[,3], 30, 30)

## 3. Plotly 3-D

library(plotly)
pal <- c("#1f77b4", "#ff7f0e")
pal1 = c( "#d62728","#2ca02c" )# colori dei due cluster

plot_ly() %>% 
  # dati originali
  add_markers(x = X[,1], y = X[,2], z = X[,3],
              color = y, colors = pal, opacity = 0.25,
              symbol = y, symbols = c("circle","square"),
              name = "Dati originali", marker = list(size = 4)) %>% 
  # punti ridotti & riportati sul piano
  add_markers(x = pred[,1], y = pred[,2], z = pred[,3],
              color = y, colors = pal,
              name = "UMAP interp.", marker = list(size = 3)) %>% 
  # superficie del piano interpolante
  add_surface(x = xg, y = yg, z = zg,
              showscale = FALSE, opacity = 0.25,
              name = "Piano UMAP") %>% 
  layout(scene = list(xaxis = list(title = "X1"),
                      yaxis = list(title = "X2"),
                      zaxis = list(title = "X3")),
         legend = list(x = 0.8, y = 0.9))


M  <- cbind(1, emb)               # 1000 × 3 : [1  u1  u2]
B  <- solve(t(M) %*% M, t(M) %*% X)[, 1:2]   # ***solo X1 e X2***
pred_xy <- M %*% B                # 1000 × 2 : coordinate (X1, X2)

## ***nuove coordinate 3-D “appiattite” ***
pred3d <- cbind(pred_xy, z = 0)   # X3 fissato a 0

## 3. Piano orizzontale di supporto ──────────────────────────────
xg <- seq(min(X[,1]), max(X[,1]), length.out = 30)
yg <- seq(min(X[,2]), max(X[,2]), length.out = 30)
xg_mat <- matrix(rep(xg, each = 30), 30, 30)
yg_mat <- matrix(rep(yg, times = 30), 30, 30)
zg_mat <- matrix(0, 30, 30)

plot_ly() %>%
  ## dati originali
  add_markers(x = X[,1], y = X[,2], z = X[,3],
              color  = y, colors = pal, opacity = 0.25,
              symbol = y, symbols = c("circle", "square"),
              name   = "Dati originali",
              marker = list(size = 4)) %>%
  ## punti UMAP proiettati sul piano X1–X2
  add_markers(x = pred3d[,1], y = pred3d[,2], z = pred3d[,3],
              color  = y, colors = pal1,
              name   = "UMAP interp. (XY)",
              marker = list(size = 3)) %>%
  ## superficie orizzontale a quota 0
  add_surface(x = xg_mat, y = yg_mat, z = zg_mat,
              showscale = FALSE, opacity = 0.20,
              name = "Piano X1–X2") %>%
  layout(scene = list(xaxis = list(title = "X1"),
                      yaxis = list(title = "X2"),
                      zaxis = list(title = "X3")),
         legend = list(x = 0.80, y = 0.90))
