##### Hypothese 3 #####

library(ARTool)
library(emmeans)
library(tidyverse)
library(readr)
library(DescTools)

#daten_sidney$Tiny <- readr::parse_number(daten_sidney$Tiny, locale = readr::locale(decimal_mark = ","))
#daten_sidney$Small <- readr::parse_number(daten_sidney$Small, locale = readr::locale(decimal_mark = ","))
#daten_sidney$Medium <- readr::parse_number(daten_sidney$Medium, locale = readr::locale(decimal_mark = ","))
#daten_sidney$Large <- readr::parse_number(daten_sidney$Large, locale = readr::locale(decimal_mark = ","))

#daten_sidney_long <- daten_sidney %>% 
#  pivot_longer(cols = c(Tiny, Small, Medium, Large), names_to = "model", values_to = "f1")

sd_daten <- daten_sidney_long %>% 
  group_by(model, optimizer) %>% 
  summarize(f1_sd = sd(f1))

sd_daten

delta <- sd_daten   %>%         
  mutate(baseline_sd = f1_sd[optimizer == "Baseline"]) %>%
  ungroup()               %>%
  filter(optimizer != "Baseline") %>%                   
  mutate(delta = baseline_sd - f1_sd)

delta_mat <- delta %>% 
  pivot_wider(id_cols = optimizer, names_from = model, values_from = delta) %>% 
  as.matrix()

delta %>% 
  pivot_wider(id_cols = optimizer, names_from = model, values_from = delta)

page_res <- PageTest(delta_mat[, c("Tiny","Small","Medium","Large")])  
print(page_res)

