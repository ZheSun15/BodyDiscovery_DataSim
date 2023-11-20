# install.packages("heatmaply")
# library(heatmaply)
library(ggplot2)
library(openxlsx)
library(jsonlite)
setwd("C:\\Users\\admin\\Downloads\\bodyDiscovery")

# set params
p_th = 0.01

predict_body <- function(data_path){
  print("starting from:")
  print(data_path)
  #ground truth me
  # 读取文本文件的每一行
  lines <- fromJSON(paste0(data_path, "/ran.json"))
  # 你可以通过索引来访问不同的行
  N<- as.numeric(lines[1])  # 第一行
  # 使用strsplit函数将第二行的逗号分隔的数字拆分为一个字符向量
  line2 <- lines[2]  # 获取第二行
  # D <- unlist(strsplit(line2, ","))  # 使用逗号分隔拆分数字
  # 将字符向量转换为数值向量
  # D <- as.numeric(D)
  D <- as.numeric(line2[[1]])
  K <-as.numeric(lines[3])  # 第三行
  Q <-as.numeric(lines[4]) #api 个数
  T=length(D)
  #MCMC parameter 
  flag_max=10000
  
  #读取 excel
  # 获取文件夹中的所有Excel文件的文件名
  excel_files <- list.files(path = data_path, pattern = "feature.*.xlsx", full.names = TRUE, all.files=FALSE)
  # 创建一个空列表来存储读取的矩阵
  matrix_list <- list()
  # 循环遍历每个Excel文件并读取数据
  for (file_path in excel_files){
    if (grepl("\\$", file_path)){  # 跳过隐藏文件
      next
    }
    # 使用read.xlsx函数读取Excel文件中的数据
    data <- read.xlsx(file_path)
    data <- data[, -1]
    # 将数据存储为矩阵，并添加到列表中
    matrix_list[[file_path]] <- as.matrix(data)
    cat("Matrix from", file_path, "has been read and stored.\n")
  }
  
  #treatment transformation
  index_0=c(1:T)[D==0]+1
  n0=sum(D==0)
  index=list()
  nx=list()
  for(q in 1:Q){index[[q]]=c(1:T)[D==q]+1
  nx[[q]]=sum(D==q)
  }
  p_list <<- list()
  effect_list<<-list()
  #tables
  for(k in 1:K){
    x=matrix_list[[k]]
    treatment=list()
    treatment_0=rowMeans(x[,index_0]-x[,index_0-1])
    p.value=matrix(0,nrow = N,ncol = Q)
    effect.value=matrix(0,nrow = N,ncol = Q)
    for(q in 1:Q){
      treatment[[q]]=rowMeans(x[,index[[q]]]-x[,index[[q]]-1])
      effect.value[,q]=treatment[[q]]-treatment_0
      #index MCMC
      index_pool<-c(index[[q]],index_0)
      flag=1
      repeat{
        try0<-sample(c(1:length(index_pool)),length(index_0))
        pseudo_index0<-index_pool[try0]
        pseudo_indext<-index_pool[-try0]
        treatment_ps0=c()
        treatment_ps0=rowMeans(x[,pseudo_index0]-x[,pseudo_index0-1])
        
        treatment_pst=c()
        treatment_pst=rowMeans(x[,pseudo_indext]-x[,pseudo_indext-1])
        effect_ps=c()
        effect_ps=treatment_pst-treatment_ps0
        
        p.value[,q]=p.value[,q]+(abs(effect_ps)>=abs(effect.value[,q]))
        
        flag=flag+1
        if(flag>flag_max){break}
      }
      ### wait to finish
      p.value[,q]=p.value[,q]/flag
      
    }
    p_list[[k]]<<-p.value
    effect_list[[k]]<<-effect.value
  }
  
  # save p list
  data_type_label <- strsplit(data_path, "/")
  data_type_label <- tail(unlist(data_type_label), 1)
  write_json(p_list, paste0(data_type_label, "_plist.json"))
  write_json(effect_list, paste0(data_type_label, "_effectlist.json"))
  
  # # Accuracy
  # # load ground truth
  # gt <- fromJSON(paste0(data_path, "/list_data.json"))
  # recall = list()
  # precision = list()
  # recall_all_ft = list()
  # precision_all_ft = list()
  # for (api_idx in 1:Q) {
  #   # 综合所有feature的结论
  #   pred_objs_all_ft <- list()
  #   for (ft_idx in 1:K) {
  #     # check api accordingly
  #     gt_objs <- gt[[ft_idx]][[api_idx]] + 1  # gt(python) starts from 0 while pred(R) starts from 1
  #     #cat("gt is:", gt_objs, "\n")
  #     pred_objs <- p_list[[ft_idx]][, api_idx]  # 取 api_idx 列
  #     pred_objs <- which(pred_objs < p_th)
  #     pred_objs_all_ft <- union(pred_objs_all_ft, pred_objs)
  #     #cat("pred is:", pred_objs, "\n")
  #     correct_ids <- intersect(pred_objs, gt_objs)
  #     #cat("correct elements: ", correct, "\n")
  #     recall <- c(recall, length(correct_ids) / length(gt_objs))
  #     #cat("recall is:", length(correct_ids) / length(gt_objs), "\n")
  #     precision <- c(precision, length(correct_ids) / length(pred_objs))
  #   }
  #   correct_id_all_ft <- intersect(pred_objs_all_ft, gt_objs)
  #   # cat("pred is:", unlist(pred_objs_all_ft), "\n")
  #   # cat("gt is:", gt_objs, "\n")
  #   recall_all_ft <- c(recall_all_ft, length(correct_id_all_ft) / length(gt_objs))
  #   precision_all_ft <- c(precision_all_ft, length(correct_id_all_ft) / length(pred_objs_all_ft))
  # }
  # 
  # return(list(recall=recall_all_ft, precision=precision_all_ft))
}

# data_root <- "D:/BIGAI/Projects/BodyDiscovery/BodyDiscovery_DataSim/data"
data_root <- "E:/BodyDiscovery_data/data_ablation_S2-S4/data"
data_types = list(
  # "3d_rotation",
  "2d_position",
  "3d_position",
  "light"
)

# recall_list <- list()
# precision_list <- list()
# p_list=list()
# effect_list=list()
for (data_type in data_types){
  # recall_list[[data_type]] <- list()
  # precision_list[[data_type]] <- list()
  data_folders <- list.dirs(data_root, full.names = TRUE)
  data_folders <- data_folders[grep(data_type, data_folders, fixed=TRUE)]
  # print(paste0(data_root, data_type))
  for (data_folder in data_folders){
    result <- predict_body(data_folder)
    # recall_list[[data_type]] <- c(recall_list[[data_type]], result$recall)
    # precision_list[[data_type]] <- c(precision_list[[data_type]], result$precision)
  }
}
print("all done.")
