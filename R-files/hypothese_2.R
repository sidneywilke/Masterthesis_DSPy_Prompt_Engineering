##### Hypothese 2 #####

library(ARTool)
library(emmeans)
library(tidyverse)
library(readr)
library(PMCMRplus)  
library(DescTools)

#daten_sidney$Tiny <- readr::parse_number(daten_sidney$Tiny, locale = readr::locale(decimal_mark = ","))
#daten_sidney$Small <- readr::parse_number(daten_sidney$Small, locale = readr::locale(decimal_mark = ","))
#daten_sidney$Medium <- readr::parse_number(daten_sidney$Medium, locale = readr::locale(decimal_mark = ","))
#daten_sidney$Large <- readr::parse_number(daten_sidney$Large, locale = readr::locale(decimal_mark = ","))

#daten_sidney_long <- daten_sidney %>% 
#  pivot_longer(cols = c(Tiny, Small, Medium, Large), names_to = "model", values_to = "f1")

delta <- daten_sidney_long   %>%                         
  group_by(run, model)       %>%
  mutate(baseline_f1 = f1[optimizer == "Baseline"]) %>% 
  ungroup()               %>%
  filter(optimizer != "Baseline") %>%                    
  mutate(delta = f1 - baseline_f1)

mean_delta <- delta %>% 
  group_by(run, model)    %>%      
  summarise(delta = mean(delta), .groups = "drop")

delta_mat <- mean_delta %>%                 
  pivot_wider(names_from = model,
              values_from = delta)          %>%
  select(Tiny, Small, Medium, Large)        %>%  
  as.matrix()

delta_list <- mean_delta %>%                 
  pivot_wider(names_from = model,
              values_from = delta)          %>%
  select(Tiny, Small, Medium, Large)        %>%  
  as.list()
jt <- JonckheereTerpstraTest(delta_list, alternative = "decreasing")
print(jt)

page_res <- PageTest(delta_mat[, c("Large","Medium","Small","Tiny")])  
print(page_res)

##### Post-hoc tests ######
delta_df <- data.frame(
  gain  = as.vector(delta_mat),                       
  model = factor(rep(colnames(delta_mat),             
                     each  = nrow(delta_mat)),
                 levels = c("Tiny","Small","Medium","Large")),
  run   = factor(rep(1:nrow(delta_mat),              
                     times = ncol(delta_mat)))
)

friedman.test(gain ~ model | run, data = delta_df)

con_res <- frdAllPairsConoverTest(
  y            = delta_df$gain,
  groups = delta_df$model,
  blocks = delta_df$run,
  p.adjust.method = "fdr"     
)

print(con_res)

p_values_post_hoc <- as.data.frame(con_res$p.value)
write_csv(p_values_post_hoc, "hypothesis_2_p_vals.csv")
