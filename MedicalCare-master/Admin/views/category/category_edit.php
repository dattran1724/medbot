<body id="page-top">

  <!-- Page Wrapper -->
  <div id="wrapper">
    <?php require_once('./views/inc/navbar.php'); ?>
    <!-- Content Wrapper -->
    <div id="content-wrapper" class="d-flex flex-column">

      <!-- Main Content -->
      <div id="content">

        <!-- Topbar -->
        <?php require_once('./views/inc/topbar.php'); ?>
        <!-- End of Topbar -->

        <!-- Begin Page Content -->
        <div class="container-fluid">
          <div class="card shadow mb-4">
            <div class="card-header py-3">
              <h6 class="m-0 font-weight-bold text-primary">Edit Category</h6>
            </div>
            <div class="card-body">
              <form action="?act=category&process=update" method="post" enctype="multipart/form-data">
                <div class="input-group mb-3">
                  <span class="input-group-text">ID: <?php echo $category['CategoryID'] ?></span>
                  <input type="hidden" name="id" class="form-control" value="<?php echo $category['CategoryID'] ?>">
                </div>
                <div class="form-group mb-3">
                  <span class="">Name:</span>
                  <input type="text" name="name" class="form-control" placeholder="Name" value="<?php echo $category['CategoryName'] ?>">
                </div>
                <div class="form-group mb-3">
                  <span>Icon:</span>
                  <img class="mb-3  bg-dark" src="../public/img/images/icons/<?php echo $category['CategoryIcon'] ?>" height="200px" width="200px">
                  <input type="file" class="form-control" name="icon" value="<?php echo $category['CategoryIcon'] ?>">
                </div>
                <div class="form-group mb-3">
                  <span>Image:</span>
                  <img class="mb-3" src="../public/img/images/category/<?php echo $category['CategoryImage'] ?>" height="200px" width="200px">
                  <input type="file" class="form-control" name="image" value="<?php echo $category['CategoryImage'] ?>">
                </div>
                <div class="form-group mb-3 text-center">
                  <button class="btn btn-lg btn-primary" type="submit">Update</button>
                </div>
              </form>
            </div>
          </div>
        </div>
        <!-- /.container-fluid -->

      </div>
      <!-- End of Main Content -->

</body>