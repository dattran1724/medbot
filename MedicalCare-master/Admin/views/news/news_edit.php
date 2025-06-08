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
              <h6 class="m-0 font-weight-bold text-primary">Edit News</h6>
            </div>
            <div class="card-body">
              <form action="?act=news&process=update" method="post" enctype="multipart/form-data">
                <div class="input-group mb-3">
                  <span class="input-group-text">ID: <?php echo $new['id'] ?></span>
                  <input type="hidden" name="id" class="form-control" value="<?php echo $new['id'] ?>">
                </div>
                <div class="form-group mb-3">
                  <span class="">Title:</span>
                  <input type="text" name="title" class="form-control" placeholder="Name" value="<?php echo $new['title'] ?>">
                </div>
                <div class="form-group mb-3">
                  <span class="">Content:</span>
                  <textarea name="content" id="contentCK"><?php echo $new['content'] ?></textarea>
                </div>
                <div class="form-group mb-3">
                  <span>Image:</span>
                  <img class="mb-3" src="../public/img/images/news/<?php echo $new['image'] ?>" height="200px" width="200px">
                  <input type="file" class="form-control" name="image" value="<?php echo $new['image'] ?>">
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
<script>
  ClassicEditor
    .create(document.querySelector('#contentCK'))
    .catch(error => {
      console.error(error);
    });
</script>