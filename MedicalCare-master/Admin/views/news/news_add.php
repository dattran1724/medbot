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
              <h6 class="m-0 font-weight-bold text-primary">New Post</h6>
            </div>
            <div class="card-body">
              <form action="?act=news&process=store" method="post" enctype="multipart/form-data">
                <div class="form-group mb-3">
                  <span class="">Title:</span>
                  <input type="text" name="title" class="form-control" placeholder="Title">
                </div>
                <div class="form-group mb-3">
                  <span class="">Content:</span>
                  <textarea name="content" id="contentCK"></textarea>
                </div>
                <div class="form-group mb-3">
                  <span>Image:</span>
                  <input type="file" class="form-control" name="image">
                </div>
                <div class="form-group mb-3 text-center">
                  <button class="btn btn-lg btn-primary" type="submit">Post</button>
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