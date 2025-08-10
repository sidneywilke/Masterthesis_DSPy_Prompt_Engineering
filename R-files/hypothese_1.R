##### Hypothese 1 ######
library(ARTool)
library(emmeans)
library(tidyverse)
library(PMCMRplus)
library(readr)

daten_sidney$Tiny <- readr::parse_number(daten_sidney$Tiny, locale = readr::locale(decimal_mark = ","))
daten_sidney$Small <- readr::parse_number(daten_sidney$Small, locale = readr::locale(decimal_mark = ","))
daten_sidney$Medium <- readr::parse_number(daten_sidney$Medium, locale = readr::locale(decimal_mark = ","))
daten_sidney$Large <- readr::parse_number(daten_sidney$Large, locale = readr::locale(decimal_mark = ","))

daten_sidney_long <- daten_sidney %>% 
  pivot_longer(cols = c(Tiny, Small, Medium, Large), names_to = "model", values_to = "f1")

daten_sidney_long <- daten_sidney_long %>%
  rename(run = lauf) %>%         
  mutate(
    optimizer = factor(optimizer),
    model     = factor(model),
    run       = factor(run)
  )

art_mod <- art(f1 ~ optimizer * model + (1 | run), data = daten_sidney_long)

aov_art_mod <- anova(art_mod)

daten_sidney_long %>%
  group_by(model) %>%
  do({
    dfm <- .
    cat("\n=== Model:", unique(dfm$model), "===\n")
    print(friedman.test(f1 ~ optimizer | run, data = dfm))
    
    conv <- with(dfm,
                 frdAllPairsConoverTest(
                   y            = f1,
                   groups       = optimizer,
                   block        = run,
                   p.adjust.method = "holm"
                 )
    )
    
    data.frame()              
  })

#### Pairwise comparisons #######

results <- daten_sidney_long %>% 
  group_by(model) %>% 
  do({
    df <- .
    base <- df %>% filter(optimizer == "baseline") %>% arrange(run)
    
    bind_rows(lapply(setdiff(unique(df$optimizer), "baseline"), function(opt) {
      opt_df <- df %>% filter(optimizer == opt) %>% arrange(run)
      d      <- opt_df$f1 - base$f1        # paired differences

      wtest   <- wilcox.test(d, exact = FALSE, correct = FALSE)
      
      data.frame(
        model      = unique(df$model),
        optimizer  = opt,
        statistic  = wtest$statistic,
        p.value    = wtest$p.value
      )
    }))
  }) %>% 
  ungroup() %>% 
  mutate(p.adj = p.adjust(p.value, method = "holm"))

print(results)

models      <- unique(as.character(daten_sidney_long$model))
optimizers  <- setdiff(unique(as.character(daten_sidney_long$optimizer)),
                       "Baseline")        # drop the baseline itself

results <- data.frame()

for (mod in models) {
  for (optim in optimizers) {
    
    base_df <- daten_sidney_long %>% 
      filter(model == mod, optimizer == "Baseline") %>% 
      arrange(run)
    
    opt_df  <- daten_sidney_long %>% 
      filter(model == mod, optimizer == optim) %>% 
      arrange(run)
    
    x <- opt_df$f1
    y <- base_df$f1          
    
    w  <- wilcox.test(x,y,
                      exact      = TRUE,
                      correct    = FALSE,   # no continuity corr. for exact test
                      alternative = "greater")
    
    res_df <- data.frame(
      model      = mod,
      optimizer  = optim,
      W          = w$statistic,
      p.value    = w$p.value,
      stringsAsFactors = FALSE
    )
    
    results <- rbind(results, res_df)
  }
}

results_fdr <- results %>%
  group_by(model) %>%
  mutate(p.adj_fdr = p.adjust(p.value, method = "fdr")) %>%
  ungroup()

p_value_table <- results_fdr %>% 
  pivot_wider(id_cols = optimizer, names_from = model, values_from = p.adj_fdr)

getwd()
write_csv(p_value_table, "~/Downloads/p_value_table.csv")
