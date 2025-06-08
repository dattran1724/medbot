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
          <!-- DataTales Example -->
          <div class="card shadow mb-4">
            <div class="card-header py-3">
              <h6 class="m-0 font-weight-bold text-primary">News</h6>
            </div>
            <div class="card-body">
              <?php if (isset($_SESSION['msg']) && $_SESSION['msg'] != null) { ?>
                <?php switch ($_SESSION['msg']) {
                  case 'del_true': ?>
                    <div class="row mb-3 alert alert-success mx-auto" style="max-width: 12rem;">
                      Delete successful!
                    </div>
                  <?php
                    break;
                  case 'del_false': ?>
                    <div class="row mb-3 alert alert-danger mx-auto" style="max-width: 12rem;">
                      Delete fail!
                    </div>
                  <?php
                    break;
                  case 'update_true': ?>
                    <div class="row mb-3 alert alert-success mx-auto" style="max-width: 12rem;">
                      Update successful!
                    </div>
                  <?php
                    break;
                  case 'update_false': ?>
                    <div class="row mb-3 alert alert-danger mx-auto" style="max-width: 12rem;">
                      Update fail!
                    </div>
                  <?php
                    break;
                  case 'add_true': ?>
                    <div class="row mb-3 alert alert-success mx-auto" style="max-width: 12rem;">
                      Add successful!
                    </div>
                  <?php
                    break;
                  case 'add_false': ?>
                    <div class="row mb-3 alert alert-danger mx-auto" style="max-width: 12rem;">
                      Add fail!
                    </div>
              <?php break;
                }
              }
              unset($_SESSION['msg']); ?>
              <a class="btn btn-primary mx-1 my-1 mb-2" href="?act=news&process=add">
                Add New Post
              </a>
              <div class="row">
                <div class="table-responsive">
                  <table class="table table-bordered text-center" id="dataTable" width="100%" cellspacing="0">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Title</th>
                        <th>Content</th>
                        <th>Image</th>
                        <th>Created At</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      <?php foreach ($news as $new) : ?>
                        <tr>
                          <td><?php echo $new['id'] ?></td>
                          <td><?php echo $new['title'] ?></td>
                          <?php $truncatedContent = htmlspecialchars(substr($new['content'], 0, 80) . '...'); ?>
                          <td><?php echo $truncatedContent ?></td>
                          <td><?php echo $new['image'] ?></td>
                          <td><?php echo $new['created_at'] ?></td>
                          <td>
                            <div class="row mx-1 my-0">
                              <a class="col-2 col-sm btn btn-sm btn-primary my-1 mx-1" href="../?act=news_detail&id=<?php echo $new['id']; ?>">View</a>
                              <a class="col-2 col-sm btn btn-sm btn-warning my-1 mx-1" href="?act=news&process=edit&id=<?php echo $new['id']; ?>">Edit</a>
                              <a class="col-2 col-sm btn btn-sm btn-danger my-1 mx-1" href="?act=news&process=delete&id=<?php echo $new['id']; ?>">Delete</a>
                            </div>
                          </td>
                        </tr>
                      <?php endforeach; ?>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

        </div>
        <!-- /.container-fluid -->

      </div>
      <!-- End of Main Content -->

</body>