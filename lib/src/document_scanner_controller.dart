import 'dart:async';

import 'package:flutter_document_scanner/src/bloc/app/app_state.dart';
import 'package:flutter_document_scanner/src/utils/image_utils.dart';

import 'bloc/app/app_bloc.dart';
import 'bloc/app/app_event.dart';

class DocumentScannerController {
  final AppBloc _appBloc = AppBloc(
    imageUtils: ImageUtils(),
  );

  AppBloc get bloc => _appBloc;

  // * Streams
  Stream<AppStatus> get statusTakePhotoPage {
    return _appBloc.stream.transform(
      StreamTransformer.fromHandlers(
        handleData: (data, sink) => sink.add(data.statusTakePhotoPage),
      ),
    );
  }

  Stream<AppStatus> get statusCropPhoto {
    return _appBloc.stream.transform(
      StreamTransformer.fromHandlers(
        handleData: (data, sink) => sink.add(data.statusCropPhoto),
      ),
    );
  }

  ///
  Future<void> takePhoto() async {
    _appBloc.add(AppPhotoTaken());
  }

  ///
  Future<void> changePage(AppPages page) async {
    _appBloc.add(AppPageChanged(page));
  }

  ///
  Future<void> cropPhoto() async {
    _appBloc.add(AppPhotoCropped());
  }
}
