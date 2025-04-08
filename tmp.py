def do_distill(rank, generators, dataloaders,optimizers,window_sizes,device):
    teacher_generator = generators[rank[0]]  # Teacher generator is ranked first
    student_generator = generators[rank[-1]]  # Student generator is ranked last
    student_optimizer = optimizers[rank[-1]]
    teacher_generator.eval()
    student_generator.train()
    #term of teacher is longer
    if window_sizes[0] > window_sizes[-1]:
        distill_dataloader = dataloaders[rank[0]]
    else:
        distill_dataloader = dataloaders[rank[-1]]
    gap = window_sizes[-0] - window_sizes[-1]
    # Distillation process: Teacher generator to Student generator
    for batch_idx, (x, y) in enumerate(distill_dataloader):

        y=y[:,-1,:]
        y=y.to(device)
        if gap>0:
            x_teacher=x
            x_student=x[:,gap:,:]

        else:
            x_teacher=x[:,(-1)*gap:,:]
            x_student=x
        x_teacher = x_teacher.to(device)
        x_student = x_student.to(device)

        # Forward pass with teacher generator
        teacher_output = teacher_generator(x_teacher).detach()

        # Forward pass with student generator
        student_output = student_generator(x_student)

        # Calculate distillation loss (MSE between teacher and student generator's outputs)
        soft_loss = F.mse_loss(student_output, teacher_output)
        hard_loss = F.mse_loss(student_output, y)
        distillation_loss = soft_loss + hard_loss

        # Backpropagate the loss and update student generator
        student_optimizer.zero_grad()
        distillation_loss.backward()
        student_optimizer.step()  # Assuming same optimizer for all generators, modify as needed
